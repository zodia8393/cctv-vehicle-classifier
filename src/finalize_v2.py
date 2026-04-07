"""K=200 클러스터 → 최종 13종 라벨 v2

규칙:
  1. 작은 bbox (< 8K): NOISE_size
  2. AUTO 클러스터 (응집도 ≥80%): 다수결 라벨 자동 적용
  3. REVIEW 클러스터 (50~80%): 다수결 + 수동 보정
  4. MIXED 클러스터 (<50%): NOISE_mixed
  5. C5 dominant: 수동 라벨 우선

수동 라벨 (Claude 검토 결과):
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
    "C5_대형화물": "T5",
    "C6_이륜차": "T13",
}

# 수동 라벨 (Claude 검토한 C5 클러스터)
MANUAL_LABELS = {
    0: "T10",   # 측면 긴 트레일러 (KMTC 컨테이너)
    94: "T10",  # 다양한 트레일러
    68: "T5",   # 대형 탑차 후면
    171: "T5",  # 덤프/카고/특수차
    145: "T8",  # 측면 긴 트레일러 (커튼)
    198: "T5",  # 대형 카고
    36: "T5",   # 야간 대형 트럭
    128: "T4",  # 중대형 탑차 측면 (4종 가까움)
}

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
    "NOISE_size": "_노이즈_작은bbox",
    "NOISE_mixed": "_노이즈_혼합클러스터",
}


def main():
    data = np.load("/workspace/CCTV차종분류_output/clusters_v2/cluster_assignments_v2.npz", allow_pickle=True)
    paths = data["paths"]
    labels = data["labels"]
    cluster_ids = data["cluster_ids"]
    n = len(paths)

    # 클러스터별 분포
    cluster_dist = defaultdict(Counter)
    for i in range(n):
        cid = int(cluster_ids[i])
        if cid >= 0:
            cluster_dist[cid][labels[i]] += 1

    # 클러스터별 라벨 결정
    cluster_to_label = {}
    auto_cnt = review_cnt = mixed_cnt = manual_cnt = 0

    for cid, dist in cluster_dist.items():
        if cid in MANUAL_LABELS:
            cluster_to_label[cid] = MANUAL_LABELS[cid]
            manual_cnt += 1
            continue

        total = sum(dist.values())
        top_cls, top_cnt = dist.most_common(1)[0]
        purity = top_cnt / total

        if purity >= 0.80:
            cluster_to_label[cid] = DEFAULT_MAP.get(top_cls, "NOISE_mixed")
            auto_cnt += 1
        elif purity >= 0.50:
            cluster_to_label[cid] = DEFAULT_MAP.get(top_cls, "NOISE_mixed")
            review_cnt += 1
        else:
            cluster_to_label[cid] = "NOISE_mixed"
            mixed_cnt += 1

    logger.info("클러스터 분류:")
    logger.info("  MANUAL: %d개", manual_cnt)
    logger.info("  AUTO (응집도 ≥80%%): %d개", auto_cnt)
    logger.info("  REVIEW (50~80%%, 다수결): %d개", review_cnt)
    logger.info("  MIXED → NOISE: %d개", mixed_cnt)

    # 최종 라벨 부여
    final_rows = []
    label_counts = Counter()
    for i in range(n):
        cid = int(cluster_ids[i])
        if cid < 0:
            label = "NOISE_size"
        else:
            label = cluster_to_label.get(cid, "NOISE_mixed")
        final_rows.append({
            "path": paths[i],
            "original_6cls": labels[i],
            "cluster_id": cid,
            "label_13cls": label,
            "label_name": CLASS_NAMES.get(label, label),
        })
        label_counts[label] += 1

    # 저장
    csv_path = Path("/workspace/CCTV차종분류_output/final_labels_v2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=final_rows[0].keys())
        w.writeheader()
        w.writerows(final_rows)
    logger.info("저장: %s", csv_path)

    # 통계
    logger.info("=" * 50)
    logger.info("최종 분포:")
    train_total = 0
    for k in sorted(label_counts.keys()):
        v = label_counts[k]
        name = CLASS_NAMES.get(k, k)
        logger.info("  %-12s %s: %d장 (%.1f%%)", k, name, v, v / n * 100)
        if not k.startswith("NOISE"):
            train_total += v
    logger.info("  학습 가능: %d장 (%.1f%%)", train_total, train_total / n * 100)


if __name__ == "__main__":
    main()
