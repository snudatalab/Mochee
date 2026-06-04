#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils import ALLOWED, canon_split, set_seed
from src.data import load_source_heads, load_target_zcache
from src.train import eval_acc_f1_loss_from_weights
from src.merge import learn_perm_alpha_bilevel_rule


def _parse_seeds(seeds: Optional[List[int]]) -> List[int]:
    if seeds and len(seeds) > 0:
        return [int(s) for s in seeds]



def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    m = float(sum(xs) / len(xs))
    v = float(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, math.sqrt(max(v, 0.0))


def _fmt_ms(m: float, s: float) -> str:
    return f"{m:.4f} ({s:.4f})"


def _fmt4(x: float) -> str:
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--encoder", type=str, default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--artifact_root", type=str, default="../model_pretrain")
    ap.add_argument("--process_root", type=str, default="../process_data")

    ap.add_argument("--sources", type=str, nargs="+", default=["crema_d", "ravdess", "mesd", "subesco"])
    ap.add_argument("--target", type=str, required=True)

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--tgt_val_ratio", type=float, default=0.15)
    ap.add_argument("--tgt_test_ratio", type=float, default=0.7)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size_head", type=int, default=32)

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    # bilevel (learn P + alpha)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--sinkhorn_iters", type=int, default=30)
    ap.add_argument("--lam_ortho", type=float, default=0.0)

    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha_lb", type=float, default=0.0)
    ap.add_argument("--meta_scale", type=float, default=5.0)

    ap.add_argument("--inner_lr", type=float, default=1e-2)
    ap.add_argument("--inner_steps", type=int, default=1)

    # job naming
    ap.add_argument("--job_name", type=str, default="P_plus_alpha")
    ap.add_argument("--seed_off", type=int, default=50001)

    args = ap.parse_args()

    sources = [canon_split(s) for s in args.sources]
    target = canon_split(args.target)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    device = torch.device(args.device)
    num_classes = len(ALLOWED)

    print(f"[load] sources={sources}")
    src_states, head_cfg_ref, enc_dim_ref = load_source_heads(
        artifact_root=args.artifact_root,
        encoder=args.encoder,
        sources=sources,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    for s in sources:
        print(f"[load] {s} ok")

    seeds = _parse_seeds(args.seeds)

    per_seed_rows: List[Dict[str, Any]] = []

    acc_list: List[float] = []
    f1_list: List[float] = []


    zroot = "N/A"

    for seed in seeds:
        # ---------------------------
        # load target zcache
        # ---------------------------
        z_tr, y_tr, z_va, y_va, z_te, y_te, zroot, ztr_path, zva_path, zte_path = load_target_zcache(
            artifact_root=args.process_root,
            encoder=args.encoder,
            target=args.target,
            val_ratio=args.tgt_val_ratio,
            test_ratio=args.tgt_test_ratio,
            seed=seed,
        )

        print(f"\n[target] {target} | seed={seed}")
        print(f"[target zcache] {zroot}")
        print(f"  train: {ztr_path}")
        print(f"  val  : {zva_path}")
        print(f"  test : {zte_path}")

        set_seed(int(seed))

        # ---------------------------
        # Learn P + alpha via bilevel (uses tr / va as-is)
        # ---------------------------
        out = learn_perm_alpha_bilevel_rule(
            src_states=src_states,
            z_tr=z_tr, y_tr=y_tr,
            z_va=z_va, y_va=y_va,
            device=device,
            steps=int(args.epochs),
            inner_lr=float(args.inner_lr),
            inner_steps=int(args.inner_steps),
            lr_P=float(args.lr),
            tau=float(args.tau),
            sinkhorn_iters=int(args.sinkhorn_iters),
            lam_ortho=float(args.lam_ortho),
            alpha_temp=float(args.temp),
            alpha_lb=float(args.alpha_lb),
            meta_scale=float(args.meta_scale),
            seed=int(seed) + int(args.seed_off),
            verbose=bool(args.verbose),
            job_name=f"{args.job_name}/seed{seed}",
        )
        W1m, b1m, W2m, b2m = out["merged_weights"]
        m = eval_acc_f1_loss_from_weights(
            W1m, b1m, W2m, b2m,
            z_te, y_te,
            device=device,
            batch_size=int(args.batch_size_head),
            num_classes=int(num_classes),
        )
        acc = float(m["acc"])
        f1 = float(m["f1"])

        per_seed_rows.append({
            "seed": int(seed),
            "acc": acc,
            "f1": f1,
        })

        acc_list.append(acc)
        f1_list.append(f1)

    # ---------------------------
    # Summary print
    # ---------------------------
    print("\n" + "=" * 140)
    print("=" * 140)
    print(f"Sources: {', '.join(sources)}")
    print(f"Target : {target}  (tgt ratios: v={args.tgt_val_ratio}, t={args.tgt_test_ratio})")
    print("-" * 140)

    print(f"{'seed':>5s} | {'C1 acc':>10s} {'C1 f1':>10s} ")
    print("-" * 140)

    for r in per_seed_rows:
        print(
            f"{r['seed']:5d} | "
            f"{_fmt4(r['acc']):>10s} {_fmt4(r['f1']):>10s} | "
        )

    acc_m, acc_s = _mean_std(acc_list)
    f1_m, f1_s = _mean_std(f1_list)

    print("-" * 140)
    print(
        f"{'mean':>5s} | "
        f"{_fmt_ms(acc_m, acc_s):>10s} {_fmt_ms(f1_m, f1_s):>10s} | "

    )
    print("=" * 140)


if __name__ == "__main__":
    main()
