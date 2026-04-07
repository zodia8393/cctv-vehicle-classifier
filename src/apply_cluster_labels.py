"""클러스터 라벨링 결과 적용

50개 클러스터에 13종 라벨 할당 → 147K 이미지에 전파
산출:
  - clustered_labels.csv (path, original_6cls, cluster_id, korean_13cls, is_noise)
  - 통계 출력
"""

import csv
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 클러스터 → 13종 라벨 매핑 (Claude 검토 결과)
# 라벨: T1~T13 (13종) + NOISE
CLUSTER_LABELS = {
    # 1종 (승용차) - 가장 큰 그룹
    0: "T1", 2: "T1", 3: "T1", 5: "T1", 6: "T1",
    9: "T1", 17: "T1", 24: "T1", 25: "T1", 27: "T1",
    29: "T1", 31: "T1", 35: "T1", 36: "T1", 37: "T1",
    38: "T1", 43: "T1", 44: "T1", 45: "T1", 49: "T1",

    # 2종 (버스) - 4개 클러스터
    11: "T2", 18: "T2", 34: "T2", 47: "T2",

    # 3종 (소형화물 — 포터/봉고)
    28: "T3",

    # 4종 (중형화물)
    1: "T4", 7: "T4", 20: "T4",

    # 5종 (대형화물 — C33은 혼재라 일단 5종, KNN 후 재검토)
    33: "T5",

    # 노이즈 - 트래킹 오류 + 고정 시설물
    4: "NOISE", 8: "NOISE", 10: "NOISE", 12: "NOISE",
    14: "NOISE", 16: "NOISE", 19: "NOISE", 21: "NOISE",
    22: "NOISE", 23: "NOISE", 26: "NOISE", 30: "NOISE",
    32: "NOISE", 39: "NOISE", 40: "NOISE", 41: "NOISE",
    42: "NOISE", 46: "NOISE",

    # 혼합 - KNN 후 재분류 필요
    13: "MIXED", 48: "MIXED",
    # 15는 야간 노이즈
    15: "NOISE",
}

# 13종 한글명
CLASS_NAMES = {
    "T1": "1종_승용차",
    "T2": "2종_버스",
    "T3": "3종_소형화물",
    "T4": "4종_중형화물",
    "T5": "5종_대형화물",
    "T6": "6종_특수화물",
    "T7": "7종_특수",
    "T8": "8종_세미트레일러",
    "T9": "9종_풀트레일러",
    "T10": "10종_세미5축",
    "T11": "11종_풀5축",
    "T12": "12종_세미6축",
    "T13": "13종_이륜차",
    "NOISE": "노이즈",
    "MIXED": "혼합_재검토",
}


def main():
    cluster_path = Path("/workspace/CCTV차종분류_output/clusters/cluster_assignments.npz")
    output_csv = Path("/workspace/CCTV차종분류_output/clustered_labels.csv")

    data = np.load(cluster_path, allow_pickle=True)
    paths = data["paths"]
    labels = data["labels"]
    cluster_ids = data["cluster_ids"]
    n = len(paths)
    logger.info("로드: %d장", n)

    # 라벨 미정 클러스터 확인
    unique_cids = np.unique(cluster_ids)
    missing = [int(c) for c in unique_cids if int(c) not in CLUSTER_LABELS]
    if missing:
        logger.warning("라벨 미할당 클러스터: %s", missing)

    # CSV 저장
    rows = []
    label_counts = {}
    for i in range(n):
        cid = int(cluster_ids[i])
        new_label = CLUSTER_LABELS.get(cid, "MIXED")
        rows.append({
            "path": paths[i],
            "original_6cls": labels[i],
            "cluster_id": cid,
            "label_13cls": new_label,
            "label_name": CLASS_NAMES.get(new_label, ""),
            "is_noise": "Y" if new_label == "NOISE" else "N",
        })
        label_counts[new_label] = label_counts.get(new_label, 0) + 1

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    logger.info("저장: %s", output_csv)
    logger.info("=" * 50)
    logger.info("13종 라벨 분포:")
    total_clean = 0
    for k in sorted(label_counts.keys()):
        v = label_counts[k]
        name = CLASS_NAMES.get(k, k)
        logger.info("  %s (%s): %d장 (%.1f%%)", k, name, v, v / n * 100)
        if k not in ("NOISE", "MIXED"):
            total_clean += v
    logger.info("  학습 가능 (노이즈/혼합 제외): %d장", total_clean)


if __name__ == "__main__":
    main()
