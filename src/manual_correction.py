"""GT 검증 결과 기반 수동 보정

규칙:
  - T1: 유지
  - T2: 유지
  - T3 → T1 (90% 승용차로 판명)
  - T4: 유지
  - T5: bbox 면적 ≥ 80,000 만 유지, 그 미만은 T4로
  - T8 → T4 (0% 정확도, 분리 실패)
  - T10: 유지 (60% 정확)
  - T13: 유지

결과 6종: T1, T2, T4, T5, T10, T13
"""

import csv
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


CLASS_NAMES = {
    "T1": "1종_승용차",
    "T2": "2종_버스",
    "T4": "4종_중형화물",
    "T5": "5종_대형화물",
    "T10": "10종_세미5축",
    "T13": "13종_이륜차",
    "NOISE_size": "_노이즈_작은bbox",
    "NOISE_mixed": "_노이즈_혼합클러스터",
    "NOISE_cleanlab": "_노이즈_cleanlab",
}

T5_MIN_AREA = 80000  # T5 유지 최소 면적


def main():
    # v3 (CL 적용 후) 라벨 로드
    v3_path = Path("/workspace/CCTV차종분류_output/final_labels_v3_clean.csv")
    rows = []
    with open(v3_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    logger.info("v3 로드: %d행", n)

    # 보정 적용
    counts_before = Counter()
    counts_after = Counter()
    transitions = Counter()

    new_rows = []
    for r in tqdm(rows, desc="보정"):
        old = r["final_label"]
        counts_before[old] += 1

        if old == "T3":
            new = "T1"  # 소형화물 → 승용차
        elif old == "T8":
            new = "T4"  # 8종 → 4종 (분리 실패)
        elif old == "T5":
            # bbox 면적 확인
            img = cv2.imread(r["path"])
            if img is None:
                new = "NOISE_size"
            else:
                h, w = img.shape[:2]
                area = w * h
                if area >= T5_MIN_AREA:
                    new = "T5"  # 진짜 대형
                else:
                    new = "T4"  # 작으면 4종
        else:
            new = old

        if old != new:
            transitions[(old, new)] += 1

        counts_after[new] += 1
        r["v4_label"] = new
        r["v4_name"] = CLASS_NAMES.get(new, new)
        new_rows.append(r)

    # CSV 저장
    out_path = Path("/workspace/CCTV차종분류_output/final_labels_v4.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=new_rows[0].keys())
        w.writeheader()
        w.writerows(new_rows)
    logger.info("저장: %s", out_path)

    # 통계
    logger.info("=" * 60)
    logger.info("변경 사항:")
    for (old, new), cnt in transitions.most_common():
        logger.info("  %s → %s: %d장", old, new, cnt)

    logger.info("=" * 60)
    logger.info("최종 v4 분포:")
    train_total = 0
    for k in sorted(counts_after.keys()):
        v = counts_after[k]
        name = CLASS_NAMES.get(k, k)
        logger.info("  %-20s %s: %d (%.1f%%)", k, name, v, v / n * 100)
        if not k.startswith("NOISE"):
            train_total += v
    logger.info("  학습 가능: %d장 (%.1f%%)", train_total, train_total / n * 100)


if __name__ == "__main__":
    main()
