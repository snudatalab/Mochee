from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.utils import extract_mlp_params


def forward_logits(W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    h = F.relu(F.linear(x, W1, b1))
    return F.linear(h, W2, b2)


def apply_hidden_perm(
    W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, P: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    W1a = P @ W1
    b1a = P @ b1
    W2a = W2 @ P.t()
    b2a = b2
    return W1a, b1a, W2a, b2a



def _sinkhorn(logits: torch.Tensor, iters: int = 15, eps: float = 1e-8) -> torch.Tensor:
    a = logits
    for _ in range(int(iters)):
        a = a - torch.logsumexp(a, dim=1, keepdim=True)
        a = a - torch.logsumexp(a, dim=0, keepdim=True)
    P = torch.exp(a)
    P = P / (P.sum(dim=1, keepdim=True) + eps)
    P = P / (P.sum(dim=0, keepdim=True) + eps)
    return P

def learn_perm_alpha_bilevel_rule(
    src_states: List[Dict[str, torch.Tensor]],
    z_tr: torch.Tensor,
    y_tr: torch.Tensor,
    z_va: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
    steps: int,
    inner_lr: float,
    inner_steps: int,
    lr_P: float,
    tau: float,
    sinkhorn_iters: int,
    lam_ortho: float,
    alpha_temp: float,   
    alpha_lb: float,
    meta_scale: float,
    seed: int,
    verbose: bool,
    learn_P: bool = True,
    use_alpha_rule: bool = True,
    job_name: str = "bilevel",
    clip_per_grad: Optional[float] = None,
    return_best: bool = False,
) -> Dict[str, Any]:

    K = len(src_states)
    if K < 2:
        raise ValueError("Need >=2 sources")
    if float(alpha_lb) < 0:
        raise ValueError("alpha_lb must be >= 0")
    if float(alpha_lb) > 0 and float(alpha_lb) * K >= 1.0:
        raise ValueError(f"alpha_lb too large: need K*alpha_lb < 1 (K={K}, lb={alpha_lb})")
    if float(tau) <= 0:
        raise ValueError("tau must be > 0")

    W1_0, _, _, _ = extract_mlp_params(src_states[0])
    H = int(W1_0.shape[0])

    z_tr = z_tr.to(device)
    y_tr = y_tr.to(device)
    z_va = z_va.to(device)
    y_va = y_va.to(device)

    W1s, b1s, W2s, b2s = [], [], [], []
    for sd in src_states:
        W1, b1, W2, b2 = extract_mlp_params(sd)
        W1s.append(W1.detach().to(device))
        b1s.append(b1.detach().to(device))
        W2s.append(W2.detach().to(device))
        b2s.append(b2.detach().to(device))

    S_list: List[torch.Tensor] = []
    optP: Optional[torch.optim.Optimizer] = None

    g_init = torch.Generator(device=device).manual_seed(int(seed))
    for _ in range(K):
        S = torch.empty((H, H), device=device, dtype=torch.float32)
        S.fill_(-2.0)
        S.diagonal().fill_(2.0)
        S.requires_grad_(True)
        S_list.append(S)
    optP = torch.optim.Adam(S_list, lr=float(lr_P))


    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    def P_of(s: int, S_override: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if not learn_P:
            return torch.eye(H, device=device, dtype=torch.float32)
        S_use = S_list[s] if S_override is None else S_override[s]
        return _sinkhorn(S_use / max(float(tau), 1e-8), iters=int(sinkhorn_iters))

    def aligned_source(s: int, P: torch.Tensor):
        # hidden-unit reindexing
        W1a = P @ W1s[s]
        b1a = P @ b1s[s]
        W2a = W2s[s] @ P.t()
        b2a = b2s[s]
        return W1a, b1a, W2a, b2a

    def build_merged(alpha_w: torch.Tensor, S_override: Optional[List[torch.Tensor]] = None):
        W1m = torch.zeros_like(W1s[0])
        b1m = torch.zeros_like(b1s[0])
        W2m = torch.zeros_like(W2s[0])
        b2m = torch.zeros_like(b2s[0])
        ortho = torch.tensor(0.0, device=device)  # still unused in your codebase
        for s in range(K):
            P = P_of(s, S_override=S_override)
            W1a, b1a, W2a, b2a = aligned_source(s, P)
            W1m = W1m + alpha_w[s] * W1a
            b1m = b1m + alpha_w[s] * b1a
            W2m = W2m + alpha_w[s] * W2a
            b2m = b2m + alpha_w[s] * b2a
        return W1m, b1m, W2m, b2m, ortho

    def forward_logits(W1, b1, W2, b2, x):
        h = F.relu(F.linear(x, W1, b1))
        return F.linear(h, W2, b2)

    P_before_list = None
    if learn_P:
        with torch.no_grad():
            P_before_list = [P_of(s).detach().cpu() for s in range(K)]

    # alpha storage across epochs
    alpha_base = torch.ones(K, device=device) / K

    pbar = tqdm(range(1, int(steps) + 1), desc=f"bilevel_alphaBase[{job_name}]", dynamic_ncols=True)

    for ep in pbar:

        alpha_delta = torch.zeros(K, device=device, dtype=torch.float32, requires_grad=True)
        base_logits = torch.log(alpha_base.clamp_min(1e-12))
        alpha_soft = F.softmax(base_logits + alpha_delta, dim=0)

        S_virtual = [S.detach().clone().requires_grad_(True) for S in S_list]
        for _ in range(int(inner_steps)):
            W1m_tr, b1m_tr, W2m_tr, b2m_tr, ortho_tr = build_merged(alpha_soft, S_override=S_virtual)
            logits_tr = forward_logits(W1m_tr, b1m_tr, W2m_tr, b2m_tr, z_tr)
            loss_tr = F.cross_entropy(logits_tr, y_tr)
            if float(lam_ortho) > 0.0:
                loss_tr = loss_tr + float(lam_ortho) * ortho_tr
            gS = torch.autograd.grad(loss_tr, S_virtual, create_graph=True, retain_graph=True)
            S_virtual = [Sv - float(inner_lr) * g for Sv, g in zip(S_virtual, gS)]

        W1m_va, b1m_va, W2m_va, b2m_va, _ = build_merged(alpha_soft, S_override=S_virtual)
        logits_va = forward_logits(W1m_va, b1m_va, W2m_va, b2m_va, z_va)
        loss_va = F.cross_entropy(logits_va, y_va)

        g_delta = torch.autograd.grad(loss_va, alpha_delta, only_inputs=True)[0]
        score = torch.clamp(-float(meta_scale) * g_delta, min=0.0)
        alpha_new = F.softmax(score, dim=0)

        if float(alpha_lb) > 0.0:
            alpha_new = torch.clamp(alpha_new, min=float(alpha_lb))
            alpha_new = alpha_new / alpha_new.sum()

        alpha_new = alpha_new.detach()

        W1m2, b1m2, W2m2, b2m2, ortho2 = build_merged(alpha_new)
        logits_tr2 = forward_logits(W1m2, b1m2, W2m2, b2m2, z_tr)
        loss_P = F.cross_entropy(logits_tr2, y_tr)

        assert optP is not None
        optP.zero_grad(set_to_none=True)
        loss_P.backward()
        if clip_per_grad is not None:
            torch.nn.utils.clip_grad_norm_(S_list, max_norm=float(clip_per_grad))
        optP.step()

        alpha_base = alpha_new

        with torch.no_grad():
            W1mv, b1mv, W2mv, b2mv, _ = build_merged(alpha_base)
            logits_va2 = forward_logits(W1mv, b1mv, W2mv, b2mv, z_va)
            va_acc = float((logits_va2.argmax(dim=-1) == y_va).float().mean().item())
            va_loss = float(F.cross_entropy(logits_va2, y_va).item())

            if verbose and (ep % 10 == 0 or ep == 1):
                a = alpha_base.detach().cpu().tolist()
                a_str = "[" + ", ".join(f"{x:.5f}" for x in a) + "]"
                print(f"[{job_name}] Epoch {ep:03d} \t | tr_loss {loss_P.item():7.4f} | "
                    f"va_loss {va_loss:8.4f} | va_acc {va_acc:8.4f} | alpha={a_str}")

    with torch.no_grad():
        last_P_list = [P_of(s).detach().cpu() for s in range(K)]
        W1_last, b1_last, W2_last, b2_last, _ = build_merged(alpha_base)
        last_merged = (W1_last.detach().cpu(), b1_last.detach().cpu(), W2_last.detach().cpu(), b2_last.detach().cpu())
        last_alpha = alpha_base.detach().cpu()

    return {
        "alpha": last_alpha,
        "P_list": last_P_list,
        "P_before_list": P_before_list,
        "merged_weights": last_merged,
    }
