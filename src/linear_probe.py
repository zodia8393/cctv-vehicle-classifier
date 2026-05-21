"""Linear probe: pretrained 백본 고정 + 7-class 헤드만 학습.

Stage 1 공정 비교: ImageNet pretrained의 "feature 품질"을 학습 후 측정.
작은 GT에서도 수 분 CPU 학습 가능 (헤드만 학습).

Usage:
    python linear_probe.py --gt-manifest ... --backend timm_mnv4 --out-weights ...
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import config
from gt_builder import CropRecord, load_manifest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_features_timm(model_id: str, images: list[np.ndarray],
                          imgsz: int = 224) -> np.ndarray:
    """timm backbone으로 feature 추출 (head 제거)."""
    import timm
    model = timm.create_model(model_id, pretrained=True, num_classes=0)  # head 제거
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = images[i:i+16]
            arr = np.stack([cv2.cvtColor(cv2.resize(img, (imgsz, imgsz)), cv2.COLOR_BGR2RGB)
                            for img in batch], axis=0).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(0, 3, 1, 2))
            tensor = (tensor - mean) / std
            f = model(tensor)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def train_probe(features: np.ndarray, labels: np.ndarray,
                n_classes: int = config.NUM_CLASSES,
                val_split: float = 0.25,
                epochs: int = 50, lr: float = 1e-2, weight_decay: float = 1e-4,
                seed: int = 0) -> tuple[float, float, nn.Module]:
    """간단한 linear probe (logistic regression via PyTorch)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(features))
    rng.shuffle(idx)
    n_val = max(1, int(len(features) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_tr = torch.from_numpy(features[train_idx]).float()
    y_tr = torch.from_numpy(labels[train_idx]).long()
    X_va = torch.from_numpy(features[val_idx]).float()
    y_va = torch.from_numpy(labels[val_idx]).long()

    d = features.shape[1]
    head = nn.Linear(d, n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    # 클래스 가중치 (불균형 완화)
    class_n = np.bincount(labels, minlength=n_classes).astype(np.float32)
    class_w = torch.from_numpy(1.0 / (class_n + 1e-3))
    class_w = class_w / class_w.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    best_val = 0.0
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        logits = head(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            tr_acc = (logits.argmax(1) == y_tr).float().mean().item()
            va_acc = (head(X_va).argmax(1) == y_va).float().mean().item()
        if va_acc > best_val:
            best_val = va_acc
        if ep % 10 == 0:
            logger.info("  ep %d: loss=%.4f tr_acc=%.2f va_acc=%.2f", ep, loss.item(), tr_acc, va_acc)
    return best_val, tr_acc, head


def per_class_acc(features: np.ndarray, labels: np.ndarray, head: nn.Module) -> dict[str, float]:
    head.eval()
    with torch.no_grad():
        preds = head(torch.from_numpy(features).float()).argmax(1).numpy()
    result = {}
    for i, cls in enumerate(config.CLASS_ORDER):
        mask = labels == i
        if mask.sum() == 0:
            result[cls] = 0.0
        else:
            result[cls] = float((preds[mask] == i).mean())
    return result


def run(gt_manifest: Path, backend: str, model_id: str, out_report: Path) -> dict:
    records = load_manifest(gt_manifest)
    labels = []
    images = []
    for r in records:
        cls = r.final_class or r.claude_class
        if cls not in config.CLASS_ORDER:
            continue
        img = cv2.imread(r.crop_path)
        if img is None:
            continue
        labels.append(config.CLASS_TO_IDX[cls])
        images.append(img)
    labels = np.array(labels)
    logger.info("probe: %d samples, %d classes", len(labels), len(set(labels)))

    logger.info("extracting features: %s (%s)", backend, model_id)
    feats = extract_features_timm(model_id, images)
    logger.info("feature dim: %d", feats.shape[1])

    best_val, tr_acc, head = train_probe(feats, labels)
    per_cls = per_class_acc(feats, labels, head)

    report = {
        "backend": backend,
        "model_id": model_id,
        "n_samples": len(labels),
        "feature_dim": int(feats.shape[1]),
        "train_acc": tr_acc,
        "val_acc": best_val,
        "per_class_acc": per_cls,
    }
    logger.info("linear probe result: val_acc=%.1f%%, train_acc=%.1f%%", best_val*100, tr_acc*100)
    for c, a in per_cls.items():
        logger.info("  %s: %.1f%%", c, a*100)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-manifest", default=str(config.GT_DIR / "manifest.jsonl"))
    parser.add_argument("--backend", required=True, choices=["timm_mnv4", "timm_efficientvit"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-report",
                        default=str(config.OUTPUT_DIR / "linear_probe_report.json"))
    args = parser.parse_args()
    run(Path(args.gt_manifest), args.backend, args.model_id, Path(args.out_report))
