"""Holdout GT 멀티 봉인셋 평가 + 종합 Release Gate (v2).

기존 eval_holdout.py 단일 매니페스트 → 멀티 매니페스트 확장.

지원 봉인셋:
  - main_test_v3 (경동, 175장)         — 기존 게이트
  - urban_blind  (시내, 150장)         — 도메인 일반화
  - night_holdout_v1 (야간, 신규 10클립) — 야간 회귀 검증

종합 게이트:
  GATE_OVERALL          ≥ 85%   (main_test_v3)
  GATE_T10              ≥ 60%   (main_test_v3)
  GATE_VAL_GT_GAP       ≤ 5%p   (val 제공 시)
  GATE_NIGHT_OVERALL    ≥ 55%   (night_holdout_v1, 신규)
  GATE_NIGHT_T1         ≥ 60%   (night_holdout_v1, 신규)
  GATE_DOMAIN_GAP       ≤ 15%p  (main vs urban, 신규)

QA Blocked 해결: B-1 (멀티 봉인셋 평가)

사용:
  python3 eval_holdout_v2.py --model round_15_mnv4_balanced.pt \
    --backend timm_mnv4 \
    --manifest main:data/holdout_gt_v3/main_test_v3_manifest.jsonl \
    --manifest urban:data/holdout_gt_v3/urban_blind_test_manifest.jsonl \
    --manifest night:data/holdout_gt_v3/night_holdout_v1_manifest.jsonl \
    --val-acc 0.714 \
    --report output/gate_report_v2.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import cv2

import config
from classifier import load_model
from eval_holdout import (
    GateResult,
    compute_f1,
    macro_f1,
    stratified_bootstrap_ci,
    mcnemar_p,
)
from gt_builder import load_manifest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 신규 게이트 임계값 (config.py에 없는 것만 정의) ──────────────────
GATE_NIGHT_OVERALL = 0.55   # 야간 봉인셋 전체 (저조도 -30%p 보정)
GATE_NIGHT_T1      = 0.60   # 야간 T1 (야간 우세 클래스)
GATE_DOMAIN_GAP    = 0.15   # 경동 vs 시내 갭


@dataclass
class MultiGateResult:
    """멀티 봉인셋 종합 평가 결과."""
    model_path: str
    timestamp: str
    backend: str

    # 봉인셋별 결과 (key: main / urban / night / ...)
    per_set: dict[str, GateResult] = field(default_factory=dict)

    # 종합 게이트
    gate_overall_pass: bool = False    # main_test_v3 ≥ 85%
    gate_t10_pass: bool = False        # main_test_v3 T10 ≥ 60%
    gate_gap_pass: bool = False        # val/GT 갭 ≤ 5%p
    gate_night_overall_pass: bool = False  # 신규
    gate_night_t1_pass: bool = False       # 신규
    gate_domain_gap_pass: bool = False     # 신규

    domain_gap: float | None = None    # main vs urban accuracy 차이

    release: bool = False
    blocked_reasons: list[str] = field(default_factory=list)


def _verify_sha256(manifest: Path, seal: Path) -> None:
    expected = seal.read_text().split()[0]
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError(
            f"SHA256 MISMATCH ({manifest.name}): expected {expected}, got {actual}"
        )
    logger.info("✓ SHA256 %s: %s", manifest.name, expected[:16])


def _load_classifier(backend: str, weights: str):
    if backend.startswith("timm"):
        return load_model(backend, "", weights_path=weights)
    return load_model(backend, weights, weights_path=weights)


def _evaluate_one(
    clf,
    manifest: Path,
    sha256_seal: Path | None,
    val_acc: float | None,
    label: str,
) -> GateResult:
    """단일 봉인셋 평가 (SHA256 검증 포함)."""
    if sha256_seal and sha256_seal.exists():
        _verify_sha256(manifest, sha256_seal)
    else:
        logger.warning("SHA256 seal missing for %s — skipping verification", label)

    records = load_manifest(manifest)
    labels = [(r.final_class or r.claude_class) for r in records]
    images = [cv2.imread(r.crop_path) for r in records]
    valid = [
        (i, l) for i, (img, l) in enumerate(zip(images, labels))
        if img is not None and l in config.CLASS_ORDER
    ]
    images = [images[i] for i, _ in valid]
    labels = [l for _, l in valid]
    logger.info("[%s] %d valid records", label, len(labels))

    preds, _ = clf.predict_batch(images)

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    overall = correct / max(len(labels), 1)

    per_cls_acc: dict[str, float] = {}
    per_cls_f1: dict[str, float] = {}
    for cls in config.CLASS_ORDER:
        idx = [i for i, l in enumerate(labels) if l == cls]
        per_cls_acc[cls] = (sum(1 for i in idx if preds[i] == cls) / len(idx)) if idx else 0.0
        per_cls_f1[cls] = compute_f1(preds, labels, cls)

    gap = (val_acc - overall) if val_acc is not None else None

    return GateResult(
        model_path=str(clf),  # placeholder — 호출자가 model_path 셋업
        overall_acc=overall,
        per_class_acc=per_cls_acc,
        per_class_f1=per_cls_f1,
        val_acc=val_acc,
        val_gt_gap=gap,
        gate_overall_pass=overall >= config.GATE_OVERALL,
        gate_t10_pass=per_cls_acc.get("T10", 0) >= config.GATE_T10,
        gate_gap_pass=(gap is None) or (gap <= config.GATE_VAL_GT_GAP),
        release=False,  # 단일 set의 release는 의미 없음, 종합 게이트로 결정
        n_holdout=len(labels),
    )


def evaluate_multi(
    model_weights: str,
    manifests: dict[str, tuple[Path, Path | None]],  # {label: (manifest, sha256_seal)}
    backend: str = "ultralytics_yolo11n",
    val_acc: float | None = None,
) -> MultiGateResult:
    """멀티 봉인셋 평가 + 종합 게이트 판정.

    Args:
        model_weights: 분류기 가중치 경로
        manifests: {'main': (path, sha_path), 'urban': (path, sha_path), ...}
        backend: 분류기 백엔드
        val_acc: 학습 시 val accuracy (gap 계산용)

    Returns:
        MultiGateResult — 종합 release 판정 + blocked_reasons
    """
    clf = _load_classifier(backend, model_weights)
    result = MultiGateResult(
        model_path=model_weights,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        backend=backend,
    )

    # 각 봉인셋 평가
    for label, (manifest, seal) in manifests.items():
        try:
            res = _evaluate_one(
                clf=clf,
                manifest=manifest,
                sha256_seal=seal,
                val_acc=val_acc if label == "main" else None,
                label=label,
            )
            res.model_path = model_weights
            result.per_set[label] = res
        except Exception as e:
            logger.error("[%s] evaluation failed: %s", label, e)
            result.blocked_reasons.append(f"{label}: {e}")

    # 종합 게이트 판정
    main = result.per_set.get("main")
    urban = result.per_set.get("urban")
    night = result.per_set.get("night")

    if main is None:
        result.blocked_reasons.append("main_test_v3 평가 실패 — release 불가")
        return result

    # 1) main_test_v3 게이트 (기존 3종)
    result.gate_overall_pass = main.gate_overall_pass
    result.gate_t10_pass = main.gate_t10_pass
    result.gate_gap_pass = main.gate_gap_pass
    if not main.gate_overall_pass:
        result.blocked_reasons.append(
            f"GATE_OVERALL: {main.overall_acc:.1%} < {config.GATE_OVERALL:.0%}"
        )
    if not main.gate_t10_pass:
        result.blocked_reasons.append(
            f"GATE_T10: {main.per_class_acc.get('T10', 0):.1%} < {config.GATE_T10:.0%}"
        )
    if not main.gate_gap_pass:
        result.blocked_reasons.append(
            f"GATE_VAL_GT_GAP: {(main.val_gt_gap or 0)*100:.2f}%p > {config.GATE_VAL_GT_GAP*100:.0f}%p"
        )

    # 2) 야간 봉인셋 게이트 (신규)
    if night is not None:
        result.gate_night_overall_pass = night.overall_acc >= GATE_NIGHT_OVERALL
        result.gate_night_t1_pass = night.per_class_acc.get("T1", 0) >= GATE_NIGHT_T1
        if not result.gate_night_overall_pass:
            result.blocked_reasons.append(
                f"GATE_NIGHT_OVERALL: {night.overall_acc:.1%} < {GATE_NIGHT_OVERALL:.0%}"
            )
        if not result.gate_night_t1_pass:
            result.blocked_reasons.append(
                f"GATE_NIGHT_T1: {night.per_class_acc.get('T1', 0):.1%} < {GATE_NIGHT_T1:.0%}"
            )
    else:
        # 야간 봉인셋 없으면 PASS 처리 (참고 지표 모드)
        result.gate_night_overall_pass = True
        result.gate_night_t1_pass = True
        logger.warning("야간 봉인셋 없음 — gate_night_* 자동 PASS (참고 지표 모드)")

    # 3) 도메인 갭 (신규: main vs urban)
    if urban is not None:
        result.domain_gap = abs(main.overall_acc - urban.overall_acc)
        result.gate_domain_gap_pass = result.domain_gap <= GATE_DOMAIN_GAP
        if not result.gate_domain_gap_pass:
            result.blocked_reasons.append(
                f"GATE_DOMAIN_GAP: {result.domain_gap*100:.1f}%p > {GATE_DOMAIN_GAP*100:.0f}%p"
            )
    else:
        result.gate_domain_gap_pass = True
        logger.warning("urban_blind 봉인셋 없음 — gate_domain_gap 자동 PASS")

    # 4) 종합 release
    result.release = (
        result.gate_overall_pass
        and result.gate_t10_pass
        and result.gate_gap_pass
        and result.gate_night_overall_pass
        and result.gate_night_t1_pass
        and result.gate_domain_gap_pass
    )

    return result


def render_report_v2(result: MultiGateResult, out_md: Path) -> None:
    """멀티 봉인셋 종합 게이트 리포트 (Markdown)."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    main = result.per_set.get("main")
    urban = result.per_set.get("urban")
    night = result.per_set.get("night")

    lines = [
        "# Release Gate Report (v2 — Multi Holdout)\n",
        f"**model**: `{result.model_path}`  ",
        f"**backend**: `{result.backend}`  ",
        f"**timestamp**: {result.timestamp}\n",
        "## 종합 게이트 판정\n",
        f"**최종**: {'🟢 RELEASE' if result.release else '🔴 BLOCKED'}\n",
    ]

    if not result.release and result.blocked_reasons:
        lines.append("### 차단 사유\n")
        for r in result.blocked_reasons:
            lines.append(f"- ❌ {r}")
        lines.append("")

    lines.extend([
        "## 게이트별 결과\n",
        "| 게이트 | 값 | 임계 | 판정 |",
        "|------|---:|---:|:---:|",
    ])

    if main:
        lines.append(
            f"| 전체 정확도 (main) | {main.overall_acc:.1%} | "
            f"{config.GATE_OVERALL:.0%} | {'✅' if result.gate_overall_pass else '❌'} |"
        )
        lines.append(
            f"| T10 정확도 (main) | {main.per_class_acc.get('T10', 0):.1%} | "
            f"{config.GATE_T10:.0%} | {'✅' if result.gate_t10_pass else '❌'} |"
        )
        if main.val_gt_gap is not None:
            lines.append(
                f"| val/GT 갭 | {main.val_gt_gap*100:+.2f}%p | "
                f"≤{config.GATE_VAL_GT_GAP*100:.0f}%p | {'✅' if result.gate_gap_pass else '❌'} |"
            )

    if night:
        lines.append(
            f"| 야간 전체 정확도 | {night.overall_acc:.1%} | "
            f"{GATE_NIGHT_OVERALL:.0%} | {'✅' if result.gate_night_overall_pass else '❌'} |"
        )
        lines.append(
            f"| 야간 T1 정확도 | {night.per_class_acc.get('T1', 0):.1%} | "
            f"{GATE_NIGHT_T1:.0%} | {'✅' if result.gate_night_t1_pass else '❌'} |"
        )

    if urban and result.domain_gap is not None:
        lines.append(
            f"| 도메인 갭 (main↔urban) | {result.domain_gap*100:.1f}%p | "
            f"≤{GATE_DOMAIN_GAP*100:.0f}%p | {'✅' if result.gate_domain_gap_pass else '❌'} |"
        )

    # 봉인셋별 상세
    lines.append("\n## 봉인셋별 상세\n")
    for label, res in result.per_set.items():
        lines.append(f"### {label} (N={res.n_holdout})\n")
        lines.append(f"- 전체 정확도: **{res.overall_acc:.1%}**")
        lines.append(f"- 클래스별 정확도/F1:")
        lines.append("\n| 클래스 | 정확도 | F1 |\n|---|---:|---:|")
        for c in config.CLASS_ORDER:
            lines.append(
                f"| {c} | {res.per_class_acc.get(c, 0):.1%} | "
                f"{res.per_class_f1.get(c, 0):.1%} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines))

    # JSON (per_set 직렬화)
    json_data = {
        "model_path": result.model_path,
        "backend": result.backend,
        "timestamp": result.timestamp,
        "release": result.release,
        "blocked_reasons": result.blocked_reasons,
        "domain_gap": result.domain_gap,
        "gates": {
            "overall_pass": result.gate_overall_pass,
            "t10_pass": result.gate_t10_pass,
            "gap_pass": result.gate_gap_pass,
            "night_overall_pass": result.gate_night_overall_pass,
            "night_t1_pass": result.gate_night_t1_pass,
            "domain_gap_pass": result.gate_domain_gap_pass,
        },
        "per_set": {k: asdict(v) for k, v in result.per_set.items()},
    }
    out_md.with_suffix(".json").write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2)
    )
    logger.info("report: %s", out_md)


def _parse_manifest_arg(arg: str) -> tuple[str, Path, Path | None]:
    """--manifest label:path[:sha_path] 파싱."""
    parts = arg.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid --manifest format: {arg}")
    label = parts[0]
    manifest = Path(parts[1])
    if len(parts) == 3:
        seal = Path(parts[2])
    else:
        # 자동 추정: <manifest>.sha256
        seal = Path(str(manifest) + ".sha256")
        if not seal.exists():
            seal = None
    return label, manifest, seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-holdout evaluation with composite release gate"
    )
    parser.add_argument("--model", required=True, help="분류기 가중치 경로")
    parser.add_argument("--backend", default="ultralytics_yolo11n")
    parser.add_argument(
        "--manifest", action="append", required=True,
        help="label:manifest_path[:sha_path] 형식. 여러 번 지정 가능. "
             "label은 'main'/'urban'/'night' 권장."
    )
    parser.add_argument("--val-acc", type=float, default=None,
                        help="학습 val accuracy (main set의 gap 계산용)")
    parser.add_argument(
        "--report", default=str(config.OUTPUT_DIR / "gate_report_v2.md"),
        help="리포트 출력 경로 (.md), .json 동시 생성"
    )
    args = parser.parse_args()

    manifests: dict[str, tuple[Path, Path | None]] = {}
    for arg in args.manifest:
        label, manifest, seal = _parse_manifest_arg(arg)
        manifests[label] = (manifest, seal)

    result = evaluate_multi(
        model_weights=args.model,
        manifests=manifests,
        backend=args.backend,
        val_acc=args.val_acc,
    )
    render_report_v2(result, Path(args.report))

    if result.release:
        logger.info("🟢 RELEASE — all gates passed")
        raise SystemExit(0)
    else:
        logger.error("🔴 BLOCKED — %d gate(s) failed", len(result.blocked_reasons))
        for r in result.blocked_reasons:
            logger.error("  - %s", r)
        raise SystemExit(1)
