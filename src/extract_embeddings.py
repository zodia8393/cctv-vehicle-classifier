"""DINOv2로 147K crop 임베딩 추출

입력:  /workspace/CCTV차종분류_output/labels/verified_clean/{C1~C6}/*.jpg
출력:  /workspace/CCTV차종분류_output/embeddings.npz
       - paths: 파일경로 (N,)
       - labels: 원본 6종 라벨 (N,)
       - embeddings: DINOv2 features (N, 384)
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLEAN_DIR = Path("/workspace/CCTV차종분류_output/labels/verified_clean")
OUTPUT = Path("/workspace/CCTV차종분류_output/embeddings.npz")
BATCH_SIZE = 32


def main():
    logger.info("DINOv2 모델 로딩...")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("device: %s", device)

    # 모든 crop 수집
    all_files = []
    all_labels = []
    for cls_dir in sorted(CLEAN_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for f in sorted(cls_dir.glob("*.jpg")):
            all_files.append(str(f))
            all_labels.append(cls_name)

    logger.info("총 %d장 처리 시작", len(all_files))
    embeddings = np.zeros((len(all_files), 384), dtype=np.float32)

    start = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), BATCH_SIZE), desc="embedding", unit="batch"):
            batch_files = all_files[i:i + BATCH_SIZE]
            images = []
            for f in batch_files:
                try:
                    img = Image.open(f).convert("RGB")
                    images.append(img)
                except Exception as e:
                    # 더미 이미지
                    images.append(Image.new("RGB", (224, 224)))

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # CLS token 사용
            feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[i:i + len(batch_files)] = feats

    elapsed = time.time() - start
    logger.info("완료: %.1f분, %.1f imgs/sec", elapsed / 60, len(all_files) / elapsed)

    # 저장
    np.savez_compressed(
        OUTPUT,
        paths=np.array(all_files),
        labels=np.array(all_labels),
        embeddings=embeddings,
    )
    logger.info("저장: %s (%.1f MB)", OUTPUT, OUTPUT.stat().st_size / 1024 / 1024)


if __name__ == "__main__":
    main()
