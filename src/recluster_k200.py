"""K=200 재클러스터링 + 작은 bbox 자동 NOISE 마킹

1. 임베딩 로드 (147,521장)
2. bbox 면적 < 20,000 사전 NOISE 마킹
3. 양호 데이터만 K=200 K-Means
4. 그리드 생성 (200개)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EMB_PATH = Path("/workspace/CCTV차종분류_output/embeddings.npz")
OUT_DIR = Path("/workspace/CCTV차종분류_output/clusters_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 200
MIN_AREA = 8000


def main():
    logger.info("임베딩 로드: %s", EMB_PATH)
    data = np.load(EMB_PATH, allow_pickle=True)
    paths = data["paths"]
    labels = data["labels"]
    embeddings = data["embeddings"]
    n = len(paths)
    logger.info("로드: %d장", n)

    # bbox 면적 계산 (이미지 크기로 측정)
    logger.info("bbox 면적 계산 중...")
    areas = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(tqdm(paths, desc="size")):
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            areas[i] = w * h

    # 작은 bbox = NOISE
    is_noise = areas < MIN_AREA
    n_small = is_noise.sum()
    logger.info("작은 bbox NOISE: %d장 (%.1f%%)", n_small, n_small / n * 100)

    # 양호 데이터만 정규화 후 클러스터링
    good_idx = np.where(~is_noise)[0]
    good_emb = embeddings[good_idx]
    good_emb = good_emb / np.maximum(np.linalg.norm(good_emb, axis=1, keepdims=True), 1e-8)
    logger.info("양호: %d장", len(good_idx))

    # UMAP 차원 축소
    logger.info("UMAP 차원 축소 (384 → 32)...")
    import umap
    reducer = umap.UMAP(n_components=32, n_neighbors=30, min_dist=0.0,
                        random_state=42, low_memory=True)
    good_reduced = reducer.fit_transform(good_emb)
    logger.info("UMAP 완료")

    # K-Means K=200
    logger.info("K-Means K=%d ...", K)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, random_state=42, n_init=10, verbose=0)
    good_cids = km.fit_predict(good_reduced)
    logger.info("클러스터링 완료")

    # 전체 cluster_ids 배열 생성 (NOISE = -1)
    cluster_ids = np.full(n, -1, dtype=np.int32)
    cluster_ids[good_idx] = good_cids

    # 저장
    np.savez_compressed(
        OUT_DIR / "cluster_assignments_v2.npz",
        paths=paths, labels=labels, cluster_ids=cluster_ids, areas=areas,
    )
    logger.info("저장: cluster_assignments_v2.npz")

    # 그리드 생성
    logger.info("그리드 생성...")
    grids_dir = OUT_DIR / "grids"
    grids_dir.mkdir(exist_ok=True)
    cell = 100
    cols = 8
    samples_per = 24

    for cid in range(K):
        member_idx = np.where(cluster_ids == cid)[0]
        if len(member_idx) == 0:
            continue

        member_labels = labels[member_idx]
        u, c = np.unique(member_labels, return_counts=True)
        dist = " ".join(f"{l.split('_')[0]}:{c}" for l, c in sorted(zip(u, c), key=lambda x: -x[1]))

        np.random.seed(cid)
        sample = member_idx[np.random.choice(len(member_idx), min(samples_per, len(member_idx)), replace=False)]

        rows_n = (len(sample) + cols - 1) // cols
        grid = np.zeros((rows_n * cell, cols * cell, 3), dtype=np.uint8)
        for i, idx in enumerate(sample):
            img = cv2.imread(paths[idx])
            if img is None:
                continue
            r, c = divmod(i, cols)
            grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(img, (cell, cell))

        header = np.zeros((30, cols * cell, 3), dtype=np.uint8)
        cv2.putText(header, f"C{cid:03d} n={len(member_idx)} | {dist}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(grids_dir / f"c{cid:03d}.jpg"), np.vstack([header, grid]))

    logger.info("완료: 200개 그리드 → %s", grids_dir)


if __name__ == "__main__":
    main()
