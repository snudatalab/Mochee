from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .utils import canon_split, build_artifact_dir, load_ckpt, try_load_ckpt, get_head_cfg, load_zcache_pt

def target_candidate_ckpt_dirs(
    artifact_root: str,
    encoder: str,
    target: str,
    src_val_ratio: float,
    src_test_ratio: float,
) -> List[str]:
    t = canon_split(target)
    cand = [
        build_artifact_dir(artifact_root, encoder, t, float(src_val_ratio), float(src_test_ratio)),
    ]
    # optional: try special ratios too
    if t == "oreau":
        cand.append(build_artifact_dir(artifact_root, encoder, t, 0.15, 0.7))
    elif t == "cafe":
        cand.append(build_artifact_dir(artifact_root, encoder, t, 0.1, 0.8))
    return cand


def load_source_heads(
    artifact_root: str,
    encoder: str,
    sources: List[str],
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any], int]:
    src_states: List[Dict[str, torch.Tensor]] = []
    head_cfg_ref: Optional[Dict[str, Any]] = None
    enc_dim_ref: Optional[int] = None

    for s in sources:
        art_dir = build_artifact_dir(artifact_root, encoder, s, float(val_ratio), float(test_ratio))
        if s in ["cafe", "oreau"]:
            art_dir = build_artifact_dir(artifact_root, encoder, s, 0.05, 0.05)
        ckpt_path = os.path.join(art_dir, "ckpt.pt")
        obj = load_ckpt(ckpt_path)

        enc_dim = int(obj.get("encoder_dim", -1))
        head_cfg = get_head_cfg(obj)

        if enc_dim_ref is None:
            enc_dim_ref = enc_dim
        if head_cfg_ref is None:
            head_cfg_ref = head_cfg

        if enc_dim != enc_dim_ref:
            raise ValueError(f"encoder_dim mismatch: {s} has {enc_dim} but expected {enc_dim_ref}")
        if head_cfg != head_cfg_ref:
            raise ValueError(f"head_cfg mismatch across sources:\n{s}:{head_cfg}\nref:{head_cfg_ref}")

        src_states.append(obj["head_state"])

    assert head_cfg_ref is not None and enc_dim_ref is not None
    return src_states, head_cfg_ref, enc_dim_ref


def try_load_target_pretrained_head(
    artifact_root: str,
    encoder: str,
    target: str,
    src_val_ratio: float,
    src_test_ratio: float,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[str], Optional[Dict[str, Any]]]:
    tgt_sd: Optional[Dict[str, torch.Tensor]] = None
    tgt_ckpt_path: Optional[str] = None
    tgt_obj: Optional[Dict[str, Any]] = None

    for d in target_candidate_ckpt_dirs(artifact_root, encoder, target, src_val_ratio, src_test_ratio):
        p = os.path.join(d, "ckpt.pt")
        obj = try_load_ckpt(p)
        if obj is not None:
            tgt_obj = obj
            tgt_sd = obj["head_state"]
            tgt_ckpt_path = p
            break
    return tgt_sd, tgt_ckpt_path, tgt_obj


def load_target_zcache(
    artifact_root: str,
    encoder: str,
    target: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, str, str]:
    t = canon_split(target)
    tgt_val_ratio = val_ratio
    tgt_test_ratio = test_ratio

    tgt_dir = build_artifact_dir(artifact_root, encoder, t, float(tgt_val_ratio), float(tgt_test_ratio))
    tgt_dir = f"{tgt_dir}_seed{int(seed)}" 
    zroot = os.path.join(tgt_dir, "zcache")

    def _find_z(split: str) -> str:
        p0 = os.path.join(zroot, f"z_{t}_{split}_fp16false_seed{int(seed)}.pt")
        p1 = os.path.join(zroot, f"z_{t}_{split}_fp16true_seed{int(seed)}.pt")
        if os.path.exists(p0):
            return p0
        if os.path.exists(p1):
            return p1
        raise FileNotFoundError(
            f"Missing zcache for {t}-{split} in {zroot} "
            f"(expected fp16false/true with _seed{int(seed)})"
        )


    ztr_path = _find_z("train")
    zva_path = _find_z("val")
    zte_path = _find_z("test")

    z_tr, y_tr, sig_tr = load_zcache_pt(ztr_path)
    z_va, y_va, sig_va = load_zcache_pt(zva_path)
    z_te, y_te, sig_te = load_zcache_pt(zte_path)

    # signature check (best-effort)
    for split, sig in [("train", sig_tr), ("val", sig_va), ("test", sig_te)]:
        if isinstance(sig, dict) and sig.get("encoder", encoder) != encoder:
            raise ValueError(f"zcache encoder mismatch on {split}: sig={sig.get('encoder')} args={encoder}")

    return z_tr, y_tr, z_va, y_va, z_te, y_te, zroot, ztr_path, zva_path, zte_path
