from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm



@torch.no_grad()
def eval_acc_f1_loss_from_weights(
    W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor,
    z: torch.Tensor, y: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Dict[str, float]:
    ds = TensorDataset(z, y)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    total_loss, total_correct, total_n = 0.0, 0, 0
    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

    W1 = W1.to(device); b1 = b1.to(device); W2 = W2.to(device); b2 = b2.to(device)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        h = F.relu(F.linear(xb, W1, b1))
        logits = F.linear(h, W2, b2)

        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item())

        pred = logits.argmax(dim=-1)
        total_correct += int((pred == yb).sum().item())
        total_n += int(yb.numel())

        idx = yb * num_classes + pred
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        cm += binc.view(num_classes, num_classes)

    if total_n == 0:
        return {"loss": 0.0, "acc": 0.0, "f1": 0.0}

    tp = cm.diag().to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_per_class = 2 * precision * recall / (precision + recall + eps)

    support = cm.sum(dim=1).to(torch.float32)
    mask = support > 0
    macro_f1 = float(f1_per_class[mask].mean().item()) if mask.any() else 0.0

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "f1": macro_f1,
    }

