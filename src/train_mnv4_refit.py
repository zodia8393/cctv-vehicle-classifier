"""R15: 2-stage fine-tune — R14 weights + 자체 924 head refit (경동 특화 보강).

전략: R14가 확보한 multi-domain feature를 유지하면서,
backbone freeze + head만 학습하여 경동 도메인에 미세 조정.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from train_mnv4_full import VehicleDataset, eval_model, FocalLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run(base_weights: str, manifest: Path, sealed_test: Path,
        out_dir: Path, epochs: int = 8, lr: float = 1e-5,
        imgsz: int = 192, batch: int = 32, freeze_backbone: bool = True,
        own_only: bool = True):
    import timm
    device = "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest, filter to own-only if specified
    all_recs = [json.loads(l) for l in manifest.read_text().splitlines()]
    test_paths = set(json.loads(l)["crop_path"]
                     for l in sealed_test.read_text().splitlines())
    if own_only:
        pool = [r for r in all_recs
                if not r.get("reviewer", "").startswith("aihub")
                and r["crop_path"] not in test_paths
                and r["final_class"] != "T13"]
    else:
        pool = [r for r in all_recs
                if r["crop_path"] not in test_paths and r["final_class"] != "T13"]

    import random
    random.Random(42).shuffle(pool)
    n_val = max(20, len(pool) // 5)
    val_recs = pool[:n_val]
    train_recs = pool[n_val:]
    logger.info("refit: train=%d val=%d (own_only=%s)", len(train_recs), len(val_recs), own_only)

    cnt = Counter(r["final_class"] for r in train_recs)
    weights = np.array([1.0 / max(cnt.get(c, 1), 1) for c in config.CLASS_ORDER],
                       dtype=np.float32)
    weights = weights / weights.sum() * config.NUM_CLASSES

    tr_ds = VehicleDataset(train_recs, imgsz=imgsz, augment=True)
    va_ds = VehicleDataset(val_recs, imgsz=imgsz, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k",
                              pretrained=False, num_classes=config.NUM_CLASSES)
    state = torch.load(base_weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)

    if freeze_backbone:
        # Freeze everything except classifier head
        for name, p in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("backbone frozen: %d / %d trainable (%.1f%%)",
                    trainable, total, trainable/total*100)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))

    best_val, best_per_cls, best_ep = 0.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in tr_ld:
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        tr_loss = loss_sum / total
        tr_acc = correct / total
        va_acc, per_cls = eval_model(model, va_ld, device)
        sched.step()
        t1 = time.time()
        logger.info("ep %02d: tr_loss=%.3f tr_acc=%.3f va_acc=%.3f (%.1fs)",
                    ep, tr_loss, tr_acc, va_acc, t1 - t0)
        if va_acc > best_val:
            best_val = va_acc
            best_per_cls = per_cls
            best_ep = ep
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    logger.info("best val %.3f @ep%d → %s", best_val, best_ep, out_dir / "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-weights", required=True)
    p.add_argument("--manifest", default=str(config.GT_DIR / "manifest.jsonl"))
    p.add_argument("--sealed-test",
                   default="/workspace/prj_cctv/pipeline/data/holdout_gt_v3/main_test_v3_manifest.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--imgsz", type=int, default=192)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--mixed", action="store_true", help="use full manifest (AI Hub included)")
    args = p.parse_args()
    run(args.base_weights, Path(args.manifest), Path(args.sealed_test), Path(args.out_dir),
        epochs=args.epochs, lr=args.lr, imgsz=args.imgsz, batch=args.batch,
        freeze_backbone=not args.no_freeze, own_only=not args.mixed)
