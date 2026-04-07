"""DINOv2 임베딩 → 클러스터링 → 클래스 자동 할당

1. 임베딩 로드
2. UMAP으로 차원 축소 (384 → 32)
3. K-Means or HDBSCAN으로 클러스터링 (K=20~50)
4. 각 클러스터의 대표 이미지 5장 추출 → 그리드 저장
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EMB_PATH = Path("/workspace/CCTV차종분류_output/embeddings.npz")
CLUSTER_DIR = Path("/workspace/CCTV차종분류_output/clusters")
N_CLUSTERS = 50  # K-Means 클러스터 수


def main():
    logger.info("임베딩 로드: %s", EMB_PATH)
    data = np.load(EMB_PATH, allow_pickle=True)
    paths = data["paths"]
    labels = data["labels"]
    embeddings = data["embeddings"]
    logger.info("로드 완료: %d장, dim=%d", len(paths), embeddings.shape[1])

    # 1. L2 정규화 (cosine similarity 효과)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.maximum(norms, 1e-8)

    # 2. UMAP 차원 축소 (선택)
    logger.info("UMAP 차원 축소 (384 → 32)...")
    import umap
    reducer = umap.UMAP(n_components=32, n_neighbors=30, min_dist=0.0, random_state=42, low_memory=True)
    embeddings_reduced = reducer.fit_transform(embeddings_norm)
    logger.info("UMAP 완료: shape=%s", embeddings_reduced.shape)

    # 3. K-Means 클러스터링
    logger.info("K-Means 클러스터링 (K=%d)...", N_CLUSTERS)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, verbose=0)
    cluster_ids = kmeans.fit_predict(embeddings_reduced)
    logger.info("클러스터링 완료")

    # 4. 클러스터별 통계
    cluster_dir = CLUSTER_DIR
    cluster_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cluster_dir / "cluster_assignments.npz",
        paths=paths,
        labels=labels,
        cluster_ids=cluster_ids,
    )
    logger.info("저장: cluster_assignments.npz")

    # 5. 각 클러스터의 대표 이미지로 그리드 생성
    logger.info("클러스터별 그리드 생성...")
    grids_dir = cluster_dir / "grids"
    grids_dir.mkdir(exist_ok=True)

    cell = 100
    cols = 8
    samples_per_cluster = 24

    for cid in range(N_CLUSTERS):
        member_idx = np.where(cluster_ids == cid)[0]
        if len(member_idx) == 0:
            continue

        # 클래스 분포
        member_labels = labels[member_idx]
        unique, counts = np.unique(member_labels, return_counts=True)
        dist_str = " ".join(f"{l.split('_')[0]}:{c}" for l, c in zip(unique, counts))

        # 대표 샘플 (random)
        np.random.seed(cid)
        sample_idx = member_idx[np.random.choice(len(member_idx), min(samples_per_cluster, len(member_idx)), replace=False)]

        # 그리드 합성
        rows_n = (len(sample_idx) + cols - 1) // cols
        grid = np.zeros((rows_n * cell, cols * cell, 3), dtype=np.uint8)
        for i, idx in enumerate(sample_idx):
            img = cv2.imread(paths[idx])
            if img is None:
                continue
            r, c = divmod(i, cols)
            grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(img, (cell, cell))

        # 헤더 추가
        header_h = 30
        header = np.zeros((header_h, cols * cell, 3), dtype=np.uint8)
        text = f"Cluster {cid:02d} | n={len(member_idx)} | {dist_str}"
        cv2.putText(header, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        full = np.vstack([header, grid])
        cv2.imwrite(str(grids_dir / f"cluster_{cid:02d}.jpg"), full)

    logger.info("완료: %s", grids_dir)
    logger.info("총 %d개 클러스터 그리드 생성", N_CLUSTERS)


if __name__ == "__main__":
    main()
