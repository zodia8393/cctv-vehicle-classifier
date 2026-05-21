"""Stage 1: 3 후보 모델 zero-shot/linear-probe 벤치마크.

200장 파일럿(클래스 균형)에서 비교:
- MobileNetV4-Conv-S   (timm_mnv4)
- EfficientViT-B0      (timm_efficientvit)
- YOLO11n-cls (warm-started v8lb_self) (ultralytics)

지표: GT 정확도, 클래스별 F1 (특히 T10), CPU 지연(batch=1), FP32 크기.
통과 조건: T10 ≥40%, 지연 <3ms → 승자 선정.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

import config
from classifier import load_model, model_size_mb
from gt_builder import CropRecord, load_manifest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class CandidateResult:
    backend: str
    model_id: str
    overall_acc: float
    t10_f1: float
    per_class_acc: dict[str, float]
    latency_ms: float
    model_size_mb: float
    pass_t10: bool
    pass_latency: bool
    verdict: str


def sample_pilot(gt_records: list[CropRecord], n: int = config.PILOT_SIZE,
                 seed: int = 0) -> list[CropRecord]:
    """클래스 균형 파일럿 샘플링."""
    rng = random.Random(seed)
    by_cls: dict[str, list[CropRecord]] = {c: [] for c in config.CLASS_ORDER}
    for r in gt_records:
        cls = r.final_class or r.claude_class
        if cls in by_cls:
            by_cls[cls].append(r)
    per_class = max(1, n // len(config.CLASS_ORDER))
    pilot = []
    for cls, recs in by_cls.items():
        rng.shuffle(recs)
        pilot.extend(recs[:per_class])
    return pilot[:n]


def evaluate_candidate(backend: str, model_id: str, weights_path: str | None,
                       pilot: list[CropRecord]) -> CandidateResult:
    """단일 후보 평가. 주의: 학습 없이 zero-shot/pretrained 상태로 평가.
    실전에서는 linear-probe 후 평가해야 공정하지만 Stage 1은 빠른 screening."""
    logger.info("evaluating candidate %s (%s)", backend, model_id)
    clf = load_model(backend, model_id, weights_path=weights_path)
    images = [cv2.imread(r.crop_path) for r in pilot]
    images = [i for i in images if i is not None]
    preds, _ = clf.predict_batch(images)

    labels = [(r.final_class or r.claude_class) for r in pilot[:len(images)]]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    overall = correct / max(len(labels), 1)

    # 클래스별 acc + T10 F1
    per_cls: dict[str, float] = {}
    for cls in config.CLASS_ORDER:
        idx = [i for i, l in enumerate(labels) if l == cls]
        if not idx:
            per_cls[cls] = 0.0
        else:
            per_cls[cls] = sum(1 for i in idx if preds[i] == cls) / len(idx)

    tp = sum(1 for p, l in zip(preds, labels) if p == "T10" and l == "T10")
    fp = sum(1 for p, l in zip(preds, labels) if p == "T10" and l != "T10")
    fn = sum(1 for p, l in zip(preds, labels) if p != "T10" and l == "T10")
    t10_f1 = 2*tp / max(2*tp + fp + fn, 1)

    lat = clf.latency_ms(images, n_iter=10) if images else float("inf")
    size = model_size_mb(weights_path) if weights_path else 0.0

    pass_t10 = per_cls.get("T10", 0.0) >= config.PILOT_T10_MIN
    pass_lat = lat < config.PILOT_LATENCY_MAX
    verdict  = "PASS" if (pass_t10 and pass_lat) else "FAIL"

    return CandidateResult(
        backend=backend, model_id=model_id,
        overall_acc=overall, t10_f1=t10_f1, per_class_acc=per_cls,
        latency_ms=lat, model_size_mb=size,
        pass_t10=pass_t10, pass_latency=pass_lat, verdict=verdict,
    )


def run_benchmark(gt_manifest: Path, out_report: Path) -> list[CandidateResult]:
    gt_records = load_manifest(gt_manifest)
    pilot = sample_pilot(gt_records)
    logger.info("pilot size: %d (balanced across %d classes)", len(pilot), len(config.CLASS_ORDER))

    results = []
    for cand in config.ZEROSHOT_CANDIDATES:
        weights = cand.get("model_id") if cand["backend"] == "ultralytics_yolo11n" else None
        try:
            r = evaluate_candidate(cand["backend"], cand["model_id"], weights, pilot)
            results.append(r)
        except Exception as e:
            logger.exception("candidate %s failed: %s", cand["backend"], e)

    # 보고서 작성
    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 1 Zero-shot 후보 비교\n",
             f"파일럿: {len(pilot)}장 (균형), 기준: T10≥{config.PILOT_T10_MIN:.0%}, 지연<{config.PILOT_LATENCY_MAX}ms\n\n",
             "| 백엔드 | 모델ID | Overall | T10 F1 | 지연(ms) | 크기(MB) | 판정 |",
             "|---|---|---:|---:|---:|---:|:---:|"]
    for r in results:
        lines.append(f"| {r.backend} | {r.model_id} | {r.overall_acc:.1%} | "
                     f"{r.t10_f1:.1%} | {r.latency_ms:.2f} | {r.model_size_mb:.1f} | {r.verdict} |")
    lines.append("\n## 클래스별 정확도\n")
    lines.append("| 백엔드 | " + " | ".join(config.CLASS_ORDER) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(config.CLASS_ORDER)) + "|")
    for r in results:
        lines.append(f"| {r.backend} | " +
                     " | ".join(f"{r.per_class_acc.get(c,0):.1%}" for c in config.CLASS_ORDER) + " |")
    out_report.write_text("\n".join(lines))

    json_report = out_report.with_suffix(".json")
    json_report.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))

    passed = [r for r in results if r.verdict == "PASS"]
    if passed:
        winner = max(passed, key=lambda r: r.overall_acc)
        logger.info("Stage 1 winner: %s (overall %.1f%%, T10 F1 %.1f%%, latency %.2fms)",
                    winner.backend, winner.overall_acc*100, winner.t10_f1*100, winner.latency_ms)
    else:
        logger.warning("No candidate passed — falling back to best overall")
        winner = max(results, key=lambda r: r.overall_acc)
        logger.warning("fallback: %s (overall %.1f%%)", winner.backend, winner.overall_acc*100)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-manifest",
                        default=str(config.GT_DIR / "manifest.jsonl"))
    parser.add_argument("--out-report",
                        default=str(config.OUTPUT_DIR / "stage1_report.md"))
    args = parser.parse_args()
    run_benchmark(Path(args.gt_manifest), Path(args.out_report))
