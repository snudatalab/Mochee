from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

ALLOWED = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
SPLIT_ALIASES = {"cafe": "cafe"}


def canon_split(name: str) -> str:
    k = (name or "").strip()
    return SPLIT_ALIASES.get(k.lower(), k.lower())


def _fmt_ratio(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def build_artifact_dir(
    artifact_root: str,
    encoder_name: str,
    ds_name: str,
    val_ratio: float,
    test_ratio: float,
) -> str:
    enc_safe = encoder_name.replace("/", "-")
    ds_safe = f"ds={canon_split(ds_name)}"
    tr_ratio = max(0.0, 1.0 - float(val_ratio) - float(test_ratio))
    ratio_tag = f"tr{_fmt_ratio(tr_ratio)}_v{_fmt_ratio(val_ratio)}_t{_fmt_ratio(test_ratio)}"
    return os.path.join(artifact_root, enc_safe, ds_safe, ratio_tag)


def load_ckpt(ckpt_path: str) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    if "head_state" not in obj:
        raise ValueError(f"Invalid ckpt format: missing head_state ({ckpt_path})")
    if "head_cfg" not in obj and "artifact_cfg" not in obj:
        raise ValueError(f"Invalid ckpt format: missing head_cfg/artifact_cfg ({ckpt_path})")
    return obj


def try_load_ckpt(ckpt_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        return None
    return load_ckpt(ckpt_path)


def get_head_cfg(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "head_cfg" in obj and obj["head_cfg"] is not None:
        return obj["head_cfg"]
    cfg = obj.get("artifact_cfg", {}).get("head", None)
    if cfg is None:
        raise ValueError("head_cfg not found in ckpt")
    return cfg


def load_zcache_pt(path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"zcache not found: {path}")
    obj = torch.load(path, map_location="cpu")
    if "z" not in obj or "y" not in obj:
        raise ValueError(f"Invalid zcache format (need keys z,y): {path}")
    z = obj["z"].float()
    y = obj["y"].long()
    sig = obj.get("sig", {})
    return z, y, sig


def uniform_average_state_dict(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(state_dicts) == 0:
        raise ValueError("No state_dicts to average")
    keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if list(sd.keys()) != keys:
            raise ValueError("State dict keys mismatch across sources (head architectures differ)")
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([sd[k].float() for sd in state_dicts], dim=0).mean(dim=0)
    return out


def extract_mlp_params(sd: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # assumes 2-layer MLP head: Linear -> ReLU -> Dropout -> Linear
    W1 = sd["net.0.weight"]
    b1 = sd["net.0.bias"]
    W2 = sd["net.3.weight"]
    b2 = sd["net.3.bias"]
    return W1, b1, W2, b2

