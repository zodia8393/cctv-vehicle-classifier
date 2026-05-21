"""Holdout GT 평가 + Release Gate.

SHA256 검증 → 예측 → 혼동행렬 → 3-조건 게이트 판정.
- overall ≥ 85%
- T10 ≥ 60%
- val/GT 갭 ≤ 5%p

모든 조건 충족 시 Release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

import config
from classifier import load_model
from gt_builder import CropRecord, load_manifest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class GateResult:
    model_path: str
    overall_acc: float
    per_class_acc: dict[str, float]
    per_class_f1: dict[str, float]
    val_acc: float | None
    val_gt_gap: float | None
    gate_overall_pass: bool
    gate_t10_pass: bool
    gate_gap_pass: bool
    release: bool
    n_holdout: int


def _load_classifier(backend: str, weights: str):
    if backend.startswith("timm"):
        return load_model(backend, "", weights_path=weights)
    return load_model(backend, weights, weights_path=weights)


def compute_f1(preds: list[str], labels: list[str], cls: str) -> float:
    tp = sum(1 for p, l in zip(preds, labels) if p == cls and l == cls)
    fp = sum(1 for p, l in zip(preds, labels) if p == cls and l != cls)
    fn = sum(1 for p, l in zip(preds, labels) if p != cls and l == cls)
    return 2*tp / max(2*tp + fp + fn, 1)


def macro_f1(preds: list[str], labels: list[str]) -> float:
    """Macro-F1 across all observed classes (unweighted mean)."""
    classes = sorted(set(labels))
    if not classes:
        return 0.0
    return sum(compute_f1(preds, labels, c) for c in classes) / len(classes)


def stratified_bootstrap_ci(
    preds: list[str], labels: list[str],
    n_iter: int = 2000, ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    """Stratified bootstrap 95% CI for overall accuracy.

    Stratifies by label class so rare-class sampling stays consistent.
    Returns (mean, low, high).
    """
    import random
    rng = random.Random(seed)
    from collections import defaultdict
    by_cls = defaultdict(list)
    for p, l in zip(preds, labels):
        by_cls[l].append(1 if p == l else 0)
    if not by_cls:
        return 0.0, 0.0, 0.0
    accs = []
    for _ in range(n_iter):
        correct, total = 0, 0
        for cls, vals in by_cls.items():
            # resample with replacement within class
            sampled = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
            correct += sum(sampled)
            total += len(sampled)
        accs.append(correct / max(total, 1))
    accs.sort()
    low = accs[int(n_iter * (1 - ci) / 2)]
    high = accs[int(n_iter * (1 + ci) / 2) - 1]
    return sum(accs) / len(accs), low, high


def mcnemar_p(preds_a: list[str], preds_b: list[str], labels: list[str]) -> float:
    """McNemar test p-value comparing two predictions on same labels.

    Returns exact binomial p-value for b (correct_only_B) vs c (correct_only_A).
    """
    from math import comb
    b = sum(1 for pa, pb, l in zip(preds_a, preds_b, labels)
            if pa != l and pb == l)  # B correct, A wrong
    c = sum(1 for pa, pb, l in zip(preds_a, preds_b, labels)
            if pa == l and pb != l)  # A correct, B wrong
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # exact two-sided binomial p-value
    p = 2 * sum(comb(n, i) * 0.5**n for i in range(k + 1))
    return min(p, 1.0)


def evaluate(
    model_weights: str,
    gt_manifest: Path,
    sha256_seal: Path,
    backend: str = "ultralytics_yolo11n",
    val_acc: float | None = None,
) -> GateResult:
    # SHA256 검증
    expected = sha256_seal.read_text().split()[0]
    actual = hashlib.sha256(gt_manifest.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError(f"GT SHA256 MISMATCH: expected {expected}, got {actual}")
    logger.info("GT SHA256 verified: %s", expected[:16])

    records = load_manifest(gt_manifest)
    labels = [(r.final_class or r.claude_class) for r in records]
    images = [cv2.imread(r.crop_path) for r in records]
    valid  = [(i, l) for i, (img, l) in enumerate(zip(images, labels))
              if img is not None and l in config.CLASS_ORDER]
    images = [images[i] for i, _ in valid]
    labels = [l for _, l in valid]
    logger.info("holdout: %d valid records", len(valid))

    clf = _load_classifier(backend, model_weights)
    preds, _ = clf.predict_batch(images)

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    overall = correct / max(len(labels), 1)

    per_cls_acc: dict[str, float] = {}
    per_cls_f1:  dict[str, float] = {}
    for cls in config.CLASS_ORDER:
        idx = [i for i, l in enumerate(labels) if l == cls]
        per_cls_acc[cls] = (sum(1 for i in idx if preds[i] == cls) / len(idx)) if idx else 0.0
        per_cls_f1[cls]  = compute_f1(preds, labels, cls)

    gap = (val_acc - overall) if val_acc is not None else None
    gate_overall = overall >= config.GATE_OVERALL
    gate_t10     = per_cls_acc.get("T10", 0) >= config.GATE_T10
    gate_gap     = (gap is None) or (gap <= config.GATE_VAL_GT_GAP)
    release      = gate_overall and gate_t10 and gate_gap

    return GateResult(
        model_path=model_weights,
        overall_acc=overall,
        per_class_acc=per_cls_acc,
        per_class_f1=per_cls_f1,
        val_acc=val_acc,
        val_gt_gap=gap,
        gate_overall_pass=gate_overall,
        gate_t10_pass=gate_t10,
        gate_gap_pass=gate_gap,
        release=release,
        n_holdout=len(labels),
    )


def render_report(res: GateResult, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Release Gate Report\n",
        f"**model**: `{res.model_path}`  ",
        f"**holdout N**: {res.n_holdout}  ",
        f"**timestamp**: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n",
        "## 게이트",
        "| 조건 | 값 | 임계 | 판정 |",
        "|------|---:|---:|:---:|",
        f"| 전체 정확도 | {res.overall_acc:.1%} | {config.GATE_OVERALL:.0%} | "
        f"{'✅' if res.gate_overall_pass else '❌'} |",
        f"| T10 정확도 | {res.per_class_acc.get('T10',0):.1%} | {config.GATE_T10:.0%} | "
        f"{'✅' if res.gate_t10_pass else '❌'} |",
    ]
    if res.val_gt_gap is not None:
        lines.append(
            f"| val/GT 갭 | {res.val_gt_gap*100:+.2f}%p | ≤{config.GATE_VAL_GT_GAP*100:.0f}%p | "
            f"{'✅' if res.gate_gap_pass else '❌'} |"
        )
    lines.append(f"\n**최종**: {'🟢 RELEASE' if res.release else '🔴 BLOCKED'}\n")
    lines.append("## 클래스별 성능\n| 클래스 | 정확도 | F1 |\n|---|---:|---:|")
    for c in config.CLASS_ORDER:
        lines.append(f"| {c} | {res.per_class_acc.get(c,0):.1%} | {res.per_class_f1.get(c,0):.1%} |")
    out_md.write_text("\n".join(lines))
    out_md.with_suffix(".json").write_text(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    logger.info("report: %s", out_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="ultralytics_yolo11n")
    parser.add_argument("--gt", default=str(config.GT_DIR / "manifest.jsonl"))
    parser.add_argument("--sha256", default=str(config.GT_DIR / "manifest.jsonl.sha256"))
    parser.add_argument("--val-acc", type=float, default=None)
    parser.add_argument("--report", default=str(config.OUTPUT_DIR / "gate_report.md"))
    args = parser.parse_args()

    res = evaluate(args.model, Path(args.gt), Path(args.sha256), args.backend, args.val_acc)
    render_report(res, Path(args.report))
    raise SystemExit(0 if res.release else 1)
