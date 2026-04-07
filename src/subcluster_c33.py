"""C33 (대형화물 8,178장) 추가 클러스터링

C33 안에서 트레일러(8~12종) vs 단차 대형(5~7종)을 분리한다.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    emb_path = Path("/workspace/CCTV차종분류_output/embeddings.npz")
    cluster_path = Path("/workspace/CCTV차종분류_output/clusters/cluster_assignments.npz")
    out_dir = Path("/workspace/CCTV차종분류_output/clusters/c33_sub")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 임베딩 + 클러스터 ID 로드
    emb = np.load(emb_path, allow_pickle=True)
    cls = np.load(cluster_path, allow_pickle=True)

    paths = emb["paths"]
    embeddings = emb["embeddings"]
    cluster_ids = cls["cluster_ids"]

    # C33만 추출
    mask = cluster_ids == 33
    sub_paths = paths[mask]
    sub_emb = embeddings[mask]
    logger.info("C33 추출: %d장", len(sub_paths))

    # L2 정규화
    sub_emb = sub_emb / np.maximum(np.linalg.norm(sub_emb, axis=1, keepdims=True), 1e-8)

    # K-Means K=15 (8~12종 분리 + 단차 + 노이즈)
    from sklearn.cluster import KMeans
    K = 15
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    sub_cids = km.fit_predict(sub_emb)
    logger.info("sub-cluster K=%d 완료", K)

    # 그리드 생성
    cell = 100
    cols = 8
    samples_per = 24

    for cid in range(K):
        idx = np.where(sub_cids == cid)[0]
        n = len(idx)
        if n == 0:
            continue
        np.random.seed(cid)
        sample_idx = idx[np.random.choice(n, min(samples_per, n), replace=False)]

        rows_n = (len(sample_idx) + cols - 1) // cols
        grid = np.zeros((rows_n * cell, cols * cell, 3), dtype=np.uint8)
        for i, k in enumerate(sample_idx):
            img = cv2.imread(sub_paths[k])
            if img is None:
                continue
            r, c = divmod(i, cols)
            grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(img, (cell, cell))

        header = np.zeros((30, cols * cell, 3), dtype=np.uint8)
        cv2.putText(header, f"C33-sub {cid:02d} | n={n}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"sub_{cid:02d}.jpg"), np.vstack([header, grid]))

    # 매핑 저장
    np.savez_compressed(out_dir / "sub_assignments.npz",
                        paths=sub_paths, sub_cluster_ids=sub_cids)
    logger.info("저장: %s", out_dir)


if __name__ == "__main__":
    main()
