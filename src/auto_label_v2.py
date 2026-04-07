"""K=200 클러스터에 자동 라벨 부여 + 응집도 분석

원본 6종(YOLO 분류)과 클러스터 응집도를 보고:
  - 응집도 80%+ 클러스터: 자동 라벨 (다수결)
  - 응집도 50~80% 클러스터: REVIEW (Claude 검토)
  - 응집도 < 50% 클러스터: MIXED

산출:
  - cluster_v2_labels.csv (cluster_id, dominant_6cls, purity, auto_label_13cls, status)
  - 검토 우선 클러스터 목록
"""

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 6종 → 13종 기본 매핑
DEFAULT_MAP = {
    "C1_승용차": "T1",
    "C2_버스": "T2",
    "C3_소형화물": "T3",
    "C4_중형화물": "T4",
    "C5_대형화물": "T5",  # 5종 (sub-cluster 필요)
    "C6_이륜차": "T13",
}


def main():
    data = np.load("/workspace/CCTV차종분류_output/clusters_v2/cluster_assignments_v2.npz", allow_pickle=True)
    paths = data["paths"]
    labels = data["labels"]
    cluster_ids = data["cluster_ids"]
    n = len(paths)

    # 클러스터별 분포 계산
    cluster_dist = defaultdict(Counter)
    for i in range(n):
        cid = int(cluster_ids[i])
        if cid >= 0:
            cluster_dist[cid][labels[i]] += 1

    rows = []
    auto_count = 0
    review_count = 0
    mixed_count = 0

    for cid in sorted(cluster_dist.keys()):
        dist = cluster_dist[cid]
        total = sum(dist.values())
        top_cls, top_cnt = dist.most_common(1)[0]
        purity = top_cnt / total

        if purity >= 0.80:
            status = "AUTO"
            auto_label = DEFAULT_MAP.get(top_cls, "MIXED")
            auto_count += 1
        elif purity >= 0.50:
            status = "REVIEW"
            auto_label = DEFAULT_MAP.get(top_cls, "MIXED")
            review_count += 1
        else:
            status = "MIXED"
            auto_label = "MIXED"
            mixed_count += 1

        rows.append({
            "cluster_id": cid,
            "n": total,
            "dominant_6cls": top_cls,
            "purity": f"{purity:.2f}",
            "auto_label_13cls": auto_label,
            "status": status,
            "dist": " ".join(f"{k.split('_')[0]}:{v}" for k, v in dist.most_common(4)),
        })

    csv_path = Path("/workspace/CCTV차종분류_output/cluster_v2_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    logger.info("=" * 60)
    logger.info("K=200 클러스터 자동 라벨 결과:")
    logger.info("  AUTO (응집도 ≥80%%): %d개", auto_count)
    logger.info("  REVIEW (50~80%%): %d개", review_count)
    logger.info("  MIXED (<50%%): %d개", mixed_count)
    logger.info("  CSV: %s", csv_path)

    # 라벨별 통계 (자동 적용 시)
    label_counts = Counter()
    noise_in_clusters = (cluster_ids == -1).sum()
    for i in range(n):
        cid = int(cluster_ids[i])
        if cid < 0:
            label_counts["NOISE_size"] += 1
            continue
        # 해당 클러스터의 자동 라벨
        for r in rows:
            if r["cluster_id"] == cid:
                label_counts[r["auto_label_13cls"]] += 1
                break

    logger.info("=" * 60)
    logger.info("자동 라벨 적용 시 분포:")
    for k in sorted(label_counts.keys()):
        v = label_counts[k]
        logger.info("  %s: %d (%.1f%%)", k, v, v / n * 100)


if __name__ == "__main__":
    main()
