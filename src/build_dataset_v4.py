"""v4 라벨로 fine-tune 학습 데이터셋 구축

YOLOv8-cls 학습용 디렉토리 구조:
  dataset_v4/
    train/
      T1_1종_승용차/
      T2_2종_버스/
      ...
    val/
      ...
"""

import csv
import logging
import shutil
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

random.seed(42)
DATASET_DIR = Path("/workspace/CCTV차종분류_output/dataset_v4")
VAL_RATIO = 0.15

# 클래스 이름 매핑
CLASS_NAMES = {
    "T1": "T1_1종_승용차",
    "T2": "T2_2종_버스",
    "T4": "T4_4종_중형화물",
    "T5": "T5_5종_대형화물",
    "T10": "T10_10종_세미5축",
    "T13": "T13_13종_이륜차",
}

# 클래스 불균형 대응: 최대 샘플 수 (T1이 압도적이라 제한)
MAX_PER_CLASS = 15000


def main():
    # v4 로드
    rows = []
    with open("/workspace/CCTV차종분류_output/final_labels_v4.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            label = r["v4_label"]
            if label.startswith("NOISE"):
                continue
            if label not in CLASS_NAMES:
                continue
            rows.append((r["path"], label))

    logger.info("학습 후보: %d장", len(rows))

    # 클래스별 그룹
    by_class = defaultdict(list)
    for path, label in rows:
        by_class[label].append(path)

    # 클래스별 샘플링 (균형)
    for cls, paths in by_class.items():
        if len(paths) > MAX_PER_CLASS:
            sampled = random.sample(paths, MAX_PER_CLASS)
            by_class[cls] = sampled
            logger.info("  %s: %d → %d (제한)", cls, len(paths), MAX_PER_CLASS)
        else:
            logger.info("  %s: %d", cls, len(paths))

    # train/val 분할 + 심링크 생성
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    total_train = total_val = 0
    for cls, paths in by_class.items():
        cls_name = CLASS_NAMES[cls]
        (train_dir / cls_name).mkdir(parents=True, exist_ok=True)
        (val_dir / cls_name).mkdir(parents=True, exist_ok=True)

        random.shuffle(paths)
        n_val = max(1, int(len(paths) * VAL_RATIO))
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        for i, p in enumerate(train_paths):
            src = Path(p)
            dst = train_dir / cls_name / f"{i:06d}_{src.name}"
            try:
                dst.symlink_to(src)
            except FileExistsError:
                pass
        for i, p in enumerate(val_paths):
            src = Path(p)
            dst = val_dir / cls_name / f"{i:06d}_{src.name}"
            try:
                dst.symlink_to(src)
            except FileExistsError:
                pass

        total_train += len(train_paths)
        total_val += len(val_paths)
        logger.info("  %s: train %d / val %d", cls_name, len(train_paths), len(val_paths))

    logger.info("=" * 50)
    logger.info("데이터셋 v4 완료: train %d / val %d", total_train, total_val)
    logger.info("경로: %s", DATASET_DIR)


if __name__ == "__main__":
    main()
