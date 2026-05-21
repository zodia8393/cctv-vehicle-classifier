"""Claude 라벨링 QA 자동 게이트.

Phase A:
  prepare → runner → 머지 → [validate] → GUI OX 검수

QA 회의 결정 (2026-04-27):
  1. X 비율 검증 (< 15% 경보, > 50% 신뢰도 낮음)
  2. 클래스 분포 ±10%p 이내 (홀드아웃 기준)
  3. Triple V2 정합성 교차검증 (불일치 분석)

이 스크립트는 OX 검수 시작 전 + 검수 완료 후 모두 실행 가능.

사용:
  python3 validate_claude_labels.py
  python3 validate_claude_labels.py --post-review  # 사용자 OX 검수 후
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 경로 ──────────────────────────────────────────────────────────────
LABELING_DIR = Path("/workspace/prj_cctv/pipeline/data/labeling_v1")
PARQUET_PATH = LABELING_DIR / "crops.parquet"
HOLDOUT_DIR = Path("/workspace/prj_cctv/pipeline/data/holdout_gt_v3")


# ── 게이트 임계값 ─────────────────────────────────────────────────────
GATE_X_RATIO_MIN = 0.15      # 사용자 X 비율 < 15% → 경보
GATE_X_RATIO_MAX = 0.50      # > 50% → subagent 신뢰도 낮음
GATE_CLASS_DIST_MAX_DIFF = 0.10  # ±10%p
GATE_TRIPLE_AGREEMENT_MIN = 0.30  # Triple V2와 30% 이상 일치 (낮으면 의심)


# ── 검증 함수들 ──────────────────────────────────────────────────────
def load_holdout_distribution() -> dict[str, float]:
    """홀드아웃 main_test_v3 클래스 분포."""
    manifest = HOLDOUT_DIR / "main_test_v3_manifest.jsonl"
    if not manifest.exists():
        logger.warning("홀드아웃 매니페스트 없음 — 기본 분포 사용")
        # 경동 23 IC 일반 분포 (대략)
        return {"T1": 0.70, "T4": 0.10, "T3": 0.07,
                "T5": 0.05, "T2": 0.05, "T10": 0.02, "T13": 0.01}

    counter = Counter()
    with manifest.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                cls = rec.get("final_class") or rec.get("claude_class")
                if cls:
                    counter[cls] += 1
            except json.JSONDecodeError:
                continue
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def validate_distribution(claude_dist: dict[str, float],
                          target_dist: dict[str, float]) -> tuple[bool, list[str]]:
    """클래스 분포 ±10%p 이내 검증."""
    issues = []
    ok = True
    for cls in set(claude_dist) | set(target_dist):
        c = claude_dist.get(cls, 0.0)
        t = target_dist.get(cls, 0.0)
        diff = abs(c - t)
        if diff > GATE_CLASS_DIST_MAX_DIFF:
            issues.append(
                f"  ❌ {cls}: Claude {c*100:.1f}% vs 홀드아웃 {t*100:.1f}% (Δ{diff*100:.1f}%p)"
            )
            ok = False
        else:
            issues.append(
                f"  ✓ {cls}: Claude {c*100:.1f}% vs 홀드아웃 {t*100:.1f}% (Δ{diff*100:.1f}%p)"
            )
    return ok, issues


def validate_triple_agreement(df) -> tuple[bool, dict]:
    """Triple V2 (current_class) vs Claude 일치율."""
    labeled = df[df["claude_class"].notna()]
    if len(labeled) == 0:
        return False, {"error": "라벨 없음"}

    agreement = (labeled["current_class"] == labeled["claude_class"]).mean()

    # 클래스별 confusion matrix
    confusion = {}
    for actual_cls in labeled["claude_class"].unique():
        sub = labeled[labeled["claude_class"] == actual_cls]
        if len(sub) == 0:
            continue
        v2_dist = sub["current_class"].value_counts(normalize=True).to_dict()
        confusion[actual_cls] = v2_dist

    return (agreement >= GATE_TRIPLE_AGREEMENT_MIN), {
        "agreement": agreement,
        "confusion": confusion,
        "n_labeled": len(labeled),
    }


def validate_x_ratio(df) -> tuple[bool, dict]:
    """사용자 OX 검수 후 X 비율 검증 (post-review only)."""
    reviewed = df[df["user_decision"].notna()]
    if len(reviewed) == 0:
        return False, {"error": "OX 검수 데이터 없음"}

    x_ratio = (reviewed["user_decision"] == "X").mean()
    return (GATE_X_RATIO_MIN <= x_ratio <= GATE_X_RATIO_MAX), {
        "x_ratio": x_ratio,
        "n_reviewed": len(reviewed),
        "x_count": int((reviewed["user_decision"] == "X").sum()),
        "o_count": int((reviewed["user_decision"] == "O").sum()),
    }


# ── 메인 ──────────────────────────────────────────────────────────────
def main(args):
    if not PARQUET_PATH.exists():
        logger.error("Parquet 없음: %s", PARQUET_PATH)
        return

    import pyarrow.parquet as pq
    table = pq.read_table(PARQUET_PATH, columns=[
        "crop_id", "current_class", "current_conf",
        "claude_class", "claude_conf", "claude_reason",
        "manual_class", "user_decision", "video_category",
    ])
    df = table.to_pandas()

    labeled = df[df["claude_class"].notna()]
    logger.info("=" * 60)
    logger.info("Claude 라벨링 QA 게이트 검증")
    logger.info("=" * 60)
    logger.info("총 크롭: %d, Claude 라벨: %d", len(df), len(labeled))

    if len(labeled) == 0:
        logger.error("Claude 라벨 없음 — runner 실행 후 재시도")
        return

    all_pass = True

    # 1. 클래스 분포 검증
    logger.info("\n[1] 클래스 분포 검증 (vs 홀드아웃)")
    claude_dist = labeled["claude_class"].value_counts(normalize=True).to_dict()
    target_dist = load_holdout_distribution()
    dist_ok, dist_issues = validate_distribution(claude_dist, target_dist)
    for line in dist_issues:
        logger.info(line)
    if dist_ok:
        logger.info("  ✅ 분포 게이트 PASS")
    else:
        logger.error("  ❌ 분포 게이트 FAIL")
        all_pass = False

    # 2. Triple V2 정합성
    logger.info("\n[2] Triple V2 정합성 (vs current_class)")
    triple_ok, triple_info = validate_triple_agreement(df)
    if "error" in triple_info:
        logger.error("  ❌ %s", triple_info["error"])
        all_pass = False
    else:
        logger.info("  Agreement: %.1f%% (기준 %.0f%%)",
                    triple_info["agreement"] * 100,
                    GATE_TRIPLE_AGREEMENT_MIN * 100)
        if triple_ok:
            logger.info("  ✅ 정합성 게이트 PASS")
        else:
            logger.error("  ❌ 정합성 게이트 FAIL — Triple V2와 일치율 너무 낮음")
            all_pass = False

        # confusion 일부 출력
        logger.info("\n  [Confusion: Claude → Triple V2 분포]")
        for cls in ["T1", "T2", "T3", "T4", "T5", "T10", "T13", "NOISE"]:
            if cls in triple_info["confusion"]:
                top3 = sorted(triple_info["confusion"][cls].items(),
                              key=lambda x: -x[1])[:3]
                top3_str = ", ".join(f"{k}={v*100:.0f}%" for k, v in top3)
                logger.info("    Claude=%-5s → Triple V2: %s", cls, top3_str)

    # 3. Confidence 분포
    logger.info("\n[3] Claude Confidence 분포")
    confs = labeled["claude_conf"].dropna()
    logger.info("  평균: %.3f", confs.mean())
    logger.info("  중앙값: %.3f", confs.median())
    logger.info("  low (<0.7): %d (%.1f%%)",
                (confs < 0.7).sum(), (confs < 0.7).mean() * 100)

    # 4. 카테고리별
    logger.info("\n[4] 카테고리별 Claude 라벨링 결과")
    for cat in ["night", "backlight", "day"]:
        sub = labeled[labeled["video_category"] == cat]
        if len(sub) == 0:
            continue
        noise_pct = (sub["claude_class"] == "NOISE").mean() * 100
        logger.info("  %s: %d장 (NOISE %.1f%%)", cat, len(sub), noise_pct)

    # 5. OX 검수 후 (선택)
    if args.post_review:
        logger.info("\n[5] OX 검수 후 X 비율 (post-review)")
        x_ok, x_info = validate_x_ratio(df)
        if "error" in x_info:
            logger.warning("  %s", x_info["error"])
        else:
            logger.info("  검수 %d장: O=%d, X=%d, X비율=%.1f%%",
                        x_info["n_reviewed"], x_info["o_count"],
                        x_info["x_count"], x_info["x_ratio"] * 100)
            if x_ok:
                logger.info("  ✅ X 비율 게이트 PASS")
            else:
                logger.error("  ❌ X 비율 게이트 FAIL")
                all_pass = False

    # 종합
    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.info("🟢 모든 게이트 PASS — GUI OX 검수 진행 가능")
    else:
        logger.error("🔴 일부 게이트 FAIL — 라벨링 재검토 필요")
    logger.info("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-review", action="store_true",
                        help="OX 검수 후 X 비율 게이트 추가")
    args = parser.parse_args()
    raise SystemExit(main(args) or 0)
