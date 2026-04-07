"""GT 검증용 500장 샘플 추출

각 클래스에서 무작위 60장씩 추출 (총 ~500장)
그리드 이미지로 합성 → Claude 시각 검증
"""

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

random.seed(42)
SAMPLES_PER_CLASS = 60
GT_DIR = Path("/workspace/CCTV차종분류_output/gt_samples")


def main():
    GT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with open("/workspace/CCTV차종분류_output/final_labels_v3_clean.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["final_label"].startswith("NOISE"):
                continue
            rows.append(r)

    # 클래스별 그룹
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["final_label"]].append(r)

    logger.info("학습 가능 클래스:")
    for k in sorted(by_class.keys()):
        logger.info("  %s: %d", k, len(by_class[k]))

    # 클래스별 샘플
    sampled = []
    for cls, items in sorted(by_class.items()):
        n = min(SAMPLES_PER_CLASS, len(items))
        chosen = random.sample(items, n)
        for c in chosen:
            c["true_label"] = ""  # Claude가 채울 컬럼
        sampled.extend(chosen)

    # CSV 저장 (Claude가 채울 GT)
    csv_path = GT_DIR / "gt_template.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "path", "final_label", "true_label", "comment"])
        w.writeheader()
        for i, r in enumerate(sampled):
            w.writerow({
                "sample_id": f"S{i:04d}",
                "path": r["path"],
                "final_label": r["final_label"],
                "true_label": "",
                "comment": "",
            })
    logger.info("GT 템플릿: %s (%d장)", csv_path, len(sampled))

    # 클래스별 그리드 생성 (Claude가 보고 채점)
    cell = 100
    cols = 10
    for cls in sorted(by_class.keys()):
        cls_samples = [s for s in sampled if s["final_label"] == cls]
        if not cls_samples:
            continue

        n = len(cls_samples)
        rows_n = (n + cols - 1) // cols
        grid = np.zeros((rows_n * cell + 30, cols * cell, 3), dtype=np.uint8)
        cv2.putText(grid, f"{cls} (n={n})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, s in enumerate(cls_samples):
            img = cv2.imread(s["path"])
            if img is None:
                continue
            r, c = divmod(i, cols)
            cell_img = cv2.resize(img, (cell, cell))
            grid[30 + r * cell:30 + (r + 1) * cell, c * cell:(c + 1) * cell] = cell_img

            # sample_id 표시
            sid_idx = sampled.index(s)
            cv2.putText(grid, f"S{sid_idx:04d}", (c * cell + 2, 30 + r * cell + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.imwrite(str(GT_DIR / f"gt_{cls}.jpg"), grid)
        logger.info("  gt_%s.jpg (n=%d)", cls, n)

    logger.info("=" * 50)
    logger.info("GT 샘플 준비 완료: %d장 / 8 클래스", len(sampled))


if __name__ == "__main__":
    main()
