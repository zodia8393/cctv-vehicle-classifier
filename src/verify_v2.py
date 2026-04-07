"""v2 라벨 검증"""

import csv
import logging
from collections import defaultdict, Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXPECTED_MAPPING = {
    "C1_승용차":   ["T1"],
    "C2_버스":     ["T2"],
    "C3_소형화물": ["T3"],
    "C4_중형화물": ["T4"],
    "C5_대형화물": ["T5", "T6", "T8", "T9", "T10", "T11", "T12"],
    "C6_이륜차":   ["T13"],
}


def main():
    rows = []
    with open("/workspace/CCTV차종분류_output/final_labels_v2.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    n = len(rows)
    logger.info("로드: %d행", n)

    # Confusion Matrix
    confusion = defaultdict(lambda: Counter())
    for r in rows:
        confusion[r["original_6cls"]][r["label_13cls"]] += 1

    logger.info("=" * 80)
    logger.info("Confusion Matrix v2 (원본 6종 → 13종)")
    all_new = sorted(set(new for c in confusion.values() for new in c.keys()))
    header = f"{'원본':15s} | " + " | ".join(f"{l[:8]:8s}" for l in all_new) + " | 합계"
    logger.info(header)
    for orig in sorted(confusion.keys()):
        counts = confusion[orig]
        total = sum(counts.values())
        line = f"{orig:15s} | " + " | ".join(f"{counts.get(l, 0):8d}" for l in all_new) + f" | {total:6d}"
        logger.info(line)

    # 일치율
    logger.info("=" * 80)
    logger.info("매핑 일치율 (예상 매핑 대비, NOISE 제외)")
    total_match = 0
    total_check = 0
    for orig, expected in EXPECTED_MAPPING.items():
        counts = confusion.get(orig, Counter())
        total_orig = sum(counts.values())
        if total_orig == 0:
            continue
        match = sum(counts[e] for e in expected)
        valid = total_orig - sum(v for k, v in counts.items() if k.startswith("NOISE"))
        match_rate = match / valid * 100 if valid > 0 else 0

        wrong = {k: v for k, v in counts.items() if k not in expected and not k.startswith("NOISE")}
        top_wrong = sorted(wrong.items(), key=lambda x: -x[1])[:3]
        wrong_str = " ".join(f"{k}:{v}" for k, v in top_wrong)

        logger.info("  %-15s 일치 %d/%d (%.1f%%) | 오류 %s",
                    orig, match, valid, match_rate, wrong_str or "-")
        total_match += match
        total_check += valid

    overall = total_match / total_check * 100 if total_check > 0 else 0
    logger.info("  ── 전체 일치율: %d/%d (%.1f%%)", total_match, total_check, overall)


if __name__ == "__main__":
    main()
