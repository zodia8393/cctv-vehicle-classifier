"""YOLOv8-cls fine-tune (v4 데이터셋)

6종 분류:
  T1 승용차, T2 버스, T4 중형화물, T5 대형화물, T10 세미5축, T13 이륜차
"""

import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    from ultralytics import YOLO

    dataset = Path("/workspace/CCTV차종분류_output/dataset_v4")
    project = Path("/workspace/CCTV차종분류_output/training_v4")

    logger.info("YOLOv8n-cls 로딩...")
    model = YOLO("/tmp/yolov8n-cls.pt")

    logger.info("학습 시작: epochs=20, imgsz=128, batch=128")
    results = model.train(
        data=str(dataset),
        epochs=20,
        imgsz=128,
        batch=128,
        project=str(project),
        name="run1",
        exist_ok=True,
        device="cpu",
        workers=8,
        patience=5,
        verbose=True,
    )

    # best 모델 복사
    best = project / "run1" / "weights" / "best.pt"
    if best.exists():
        dst = Path("/workspace/CCTV차종분류_output/cls_v4_best.pt")
        shutil.copy2(best, dst)
        logger.info("best 모델 저장: %s", dst)


if __name__ == "__main__":
    main()
