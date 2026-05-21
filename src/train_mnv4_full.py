"""MobileNetV4 full fine-tune (timm 기반, CPU).

Ultralytics 백본이 아닌 timm MobileNetV4-Conv-S로 6-class 분류기 전체 학습.
linear probe에서 84.6% 관찰됨 → full fine-tune이 더 개선되는지 검증.

전략:
- Data augmentation 활성 (hflip, colorjitter, randomresizedcrop)
- 클래스 가중 손실 (rare class upweight)
- AdamW, cosine LR, weight decay
- Sealed test set은 학습 전혀 참조 안 함
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from gt_builder import CropRecord

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Dataset ─────────────────────────────────────────────────────────

class VehicleDataset(Dataset):
    def __init__(self, records: list[dict], imgsz: int = 128, augment: bool = False):
        self.records = records
        self.imgsz = imgsz
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.records)

    def _augment(self, img):
        # Horizontal flip
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
        # Random resized crop (± 20%)
        if np.random.rand() < 0.5:
            h, w = img.shape[:2]
            s = np.random.uniform(0.8, 1.0)
            nh, nw = int(h*s), int(w*s)
            y0 = np.random.randint(0, h - nh + 1)
            x0 = np.random.randint(0, w - nw + 1)
            img = img[y0:y0+nh, x0:x0+nw]
        # ColorJitter
        if np.random.rand() < 0.5:
            img = img.astype(np.float32)
            b = np.random.uniform(0.8, 1.2)
            img = np.clip(img * b, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        r = self.records[idx]
        img = cv2.imread(r["crop_path"])
        if img is None:
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        if self.augment:
            img = self._augment(img)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        label = config.CLASS_TO_IDX[r["final_class"]]
        return img, label


# ── Train ──────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss: (1-p_t)^gamma * CE(y, logits) with optional class weight."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def mixup_batch(x, y, num_classes: int, alpha: float = 0.2):
    """MixUp: mix pairs within batch with Beta(α, α) weight."""
    if alpha <= 0:
        return x, nn.functional.one_hot(y, num_classes).float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_oh = nn.functional.one_hot(y, num_classes).float()
    y_mix = lam * y_oh + (1 - lam) * y_oh[idx]
    return x_mix, y_mix


def train_epoch(model, loader, opt, loss_fn, device, mixup_alpha: float = 0.0,
                num_classes: int = 7):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        if mixup_alpha > 0:
            x_m, y_m = mixup_batch(x, y, num_classes=num_classes, alpha=mixup_alpha)
            logits = model(x_m)
            # soft-target CE
            log_p = nn.functional.log_softmax(logits, dim=1)
            loss = -(y_m * log_p).sum(dim=1).mean()
            # hard accuracy on original label
            correct += (logits.argmax(1) == y).sum().item()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            correct += (logits.argmax(1) == y).sum().item()
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, correct = 0, 0
    per_cls_total = np.zeros(config.NUM_CLASSES, dtype=int)
    per_cls_correct = np.zeros(config.NUM_CLASSES, dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        for i in range(config.NUM_CLASSES):
            mask = (y == i)
            per_cls_total[i] += mask.sum().item()
            per_cls_correct[i] += ((preds == y) & mask).sum().item()
    per_cls_acc = np.where(per_cls_total > 0,
                           per_cls_correct / np.maximum(per_cls_total, 1), 0.0)
    return correct / total, per_cls_acc


def run(
    manifest: Path, sealed_test_manifest: Path,
    out_dir: Path, epochs: int = 30, lr: float = 5e-4,
    imgsz: int = 128, batch: int = 32,
    model_id: str = "mobilenetv4_conv_small.e2400_r224_in1k",
    use_focal: bool = False, focal_gamma: float = 2.0,
    mixup_alpha: float = 0.0,
    swa_start: int | None = None,  # epoch to start SWA (None = off)
    external_t_only: bool = False,  # True: AI Hub T10/T13만 선택 (R12 전략)
):
    import timm
    device = "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all labeled, split out sealed test
    all_recs = [json.loads(l) for l in manifest.read_text().splitlines()]
    test_paths = set(json.loads(l)["crop_path"]
                     for l in sealed_test_manifest.read_text().splitlines())
    if external_t_only:
        # R12 strategy: own + AI Hub (T10/T13 only)
        filtered = []
        for r in all_recs:
            if r.get("reviewer", "").startswith("aihub_165"):
                if r["final_class"] in ("T10", "T13"):
                    filtered.append(r)
            else:
                filtered.append(r)
        all_recs = filtered
    pool = [r for r in all_recs
            if r["crop_path"] not in test_paths and r["final_class"] != "T13"]
    import random
    random.Random(42).shuffle(pool)
    n_val = max(20, len(pool) // 5)
    val_recs = pool[:n_val]
    train_recs = pool[n_val:]
    logger.info("train=%d val=%d (T13 excluded), sealed_test=%d",
                len(train_recs), len(val_recs), len(test_paths))

    # Class weights
    cnt = Counter(r["final_class"] for r in train_recs)
    weights = np.array([1.0 / max(cnt.get(c, 1), 1) for c in config.CLASS_ORDER],
                       dtype=np.float32)
    weights = weights / weights.sum() * config.NUM_CLASSES
    logger.info("class weights: %s", dict(zip(config.CLASS_ORDER, weights.round(2))))

    # Datasets / loaders
    tr_ds = VehicleDataset(train_recs, imgsz=imgsz, augment=True)
    va_ds = VehicleDataset(val_recs, imgsz=imgsz, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)

    # Model
    model = timm.create_model(model_id, pretrained=True, num_classes=config.NUM_CLASSES)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if use_focal:
        loss_fn = FocalLoss(gamma=focal_gamma, weight=torch.from_numpy(weights))
        logger.info("loss: FocalLoss(γ=%.1f)+weighted", focal_gamma)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
        logger.info("loss: weighted CE")
    if mixup_alpha > 0:
        logger.info("mixup α=%.2f", mixup_alpha)

    # SWA setup
    swa_model = None
    if swa_start is not None and swa_start < epochs:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_sched = SWALR(opt, swa_lr=lr * 0.5)
        logger.info("SWA enabled from epoch %d", swa_start)

    best_val, best_per_cls, best_ep = 0.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_ld, opt, loss_fn, device,
                                       mixup_alpha=mixup_alpha,
                                       num_classes=config.NUM_CLASSES)
        va_acc, per_cls = eval_model(model, va_ld, device)
        if swa_model is not None and ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()
        t1 = time.time()
        logger.info("ep %02d: tr_loss=%.3f tr_acc=%.3f va_acc=%.3f (%.1fs)",
                    ep, tr_loss, tr_acc, va_acc, t1 - t0)
        if va_acc > best_val:
            best_val = va_acc
            best_per_cls = per_cls
            best_ep = ep
            torch.save(model.state_dict(), out_dir / "best.pt")

    # Save final + SWA
    torch.save(model.state_dict(), out_dir / "last.pt")
    logger.info("best val %.3f @ep%d, saved: %s", best_val, best_ep, out_dir / "best.pt")
    if swa_model is not None:
        # BN stats update for SWA
        try:
            from torch.optim.swa_utils import update_bn
            update_bn(tr_ld, swa_model, device=device)
        except Exception as e:
            logger.warning("SWA BN update failed: %s", e)
        swa_acc, swa_per_cls = eval_model(swa_model, va_ld, device)
        # Save SWA weights (unwrap from AveragedModel)
        torch.save(swa_model.module.state_dict(), out_dir / "swa.pt")
        logger.info("SWA val %.3f, saved: %s", swa_acc, out_dir / "swa.pt")
        if swa_acc > best_val:
            logger.info("🏆 SWA beats best by %.2f%%p, promoting to best.pt",
                        (swa_acc - best_val) * 100)
            torch.save(swa_model.module.state_dict(), out_dir / "best.pt")
    if best_per_cls is not None:
        logger.info("best per-class: %s",
                    {c: f"{a:.2f}" for c, a in zip(config.CLASS_ORDER, best_per_cls)})
    return out_dir / "best.pt", best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(config.GT_DIR / "manifest.jsonl"))
    parser.add_argument("--sealed-test",
                        default=str(config.ROUNDS_DIR / "round_4/test_manifest.jsonl"))
    parser.add_argument("--out-dir", default=str(config.ROUNDS_DIR / "round_6_mnv4"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--focal", action="store_true", help="Use Focal Loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0=off)")
    parser.add_argument("--swa-start", type=int, default=None,
                        help="Epoch to start SWA averaging")
    parser.add_argument("--external-t-only", action="store_true",
                        help="AI Hub: use only T10/T13 (R12 strategy)")
    args = parser.parse_args()
    run(Path(args.manifest), Path(args.sealed_test), Path(args.out_dir),
        epochs=args.epochs, lr=args.lr, imgsz=args.imgsz, batch=args.batch,
        use_focal=args.focal, focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup, swa_start=args.swa_start,
        external_t_only=args.external_t_only)
