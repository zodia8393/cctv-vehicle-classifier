"""클러스터 + sub-cluster 결과를 통합하여 최종 13종 라벨 생성

산출:
  - final_labels.csv: path, label_13cls, label_name, source
  - verified_13cls/{1종~13종, NOISE}/*.jpg: 학습용 디렉토리 구조
"""

import csv
import logging
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 50개 클러스터 → 13종
CLUSTER_LABELS = {
    # 1종 승용차
    0: "T1", 2: "T1", 3: "T1", 5: "T1", 6: "T1",
    9: "T1", 17: "T1", 24: "T1", 25: "T1", 27: "T1",
    29: "T1", 31: "T1", 35: "T1", 36: "T1", 37: "T1",
    38: "T1", 43: "T1", 44: "T1", 45: "T1", 49: "T1",
    # 2종 버스
    11: "T2", 18: "T2", 34: "T2", 47: "T2",
    # 3종 소형화물
    28: "T3",
    # 4종 중형화물
    1: "T4", 7: "T4", 20: "T4",
    # 5종 대형화물 (C33은 sub로 세분화)
    33: "SUB33",  # 특수 마커
    # 노이즈
    4: "NOISE", 8: "NOISE", 10: "NOISE", 12: "NOISE",
    14: "NOISE", 15: "NOISE", 16: "NOISE", 19: "NOISE",
    21: "NOISE", 22: "NOISE", 23: "NOISE", 26: "NOISE",
    30: "NOISE", 32: "NOISE", 39: "NOISE", 40: "NOISE",
    41: "NOISE", 42: "NOISE", 46: "NOISE",
    # 혼합
    13: "MIXED", 48: "MIXED",
}

# C33 15개 sub → 13종
SUB33_LABELS = {
    0: "T10",   # 트레일러 측면
    1: "T10",   # 컨테이너 트레일러 (5축)
    2: "T4",    # 중형 탑차
    3: "T4",    # 중형 탑차
    4: "T4",    # 탑차 후면
    5: "T8",    # 트레일러 측면
    6: "MIXED", # 혼합
    7: "NOISE", # 트래킹 오류 (반복)
    8: "NOISE", # 트래킹 오류
    9: "T5",    # 덤프/대형 단차
    10: "T6",   # 레미콘/탱크로리
    11: "T3",   # 소형 탑차
    12: "T4",   # 중형 탑차
    13: "MIXED", # 혼합
    14: "NOISE", # 광고 탑차 반복
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
    "NOISE": "_노이즈",
    "MIXED": "_혼합",
}


def main():
    cluster_path = Path("/workspace/CCTV차종분류_output/clusters/cluster_assignments.npz")
    sub_path = Path("/workspace/CCTV차종분류_output/clusters/c33_sub/sub_assignments.npz")

    data = np.load(cluster_path, allow_pickle=True)
    paths = data["paths"]
    cluster_ids = data["cluster_ids"]
    n = len(paths)

    # C33 sub 매핑 생성: path → sub_cluster_id
    sub_data = np.load(sub_path, allow_pickle=True)
    sub_paths = sub_data["paths"]
    sub_cids = sub_data["sub_cluster_ids"]
    sub_map = {p: int(c) for p, c in zip(sub_paths, sub_cids)}

    # 최종 라벨 부여
    final_labels = []
    for i in range(n):
        cid = int(cluster_ids[i])
        path = paths[i]
        marker = CLUSTER_LABELS.get(cid, "MIXED")

        if marker == "SUB33":
            sub_cid = sub_map.get(path, -1)
            label = SUB33_LABELS.get(sub_cid, "MIXED")
            source = f"sub33-{sub_cid:02d}"
        else:
            label = marker
            source = f"c{cid:02d}"

        final_labels.append({
            "path": path,
            "cluster_id": cid,
            "label_13cls": label,
            "label_name": CLASS_NAMES.get(label, label),
            "source": source,
        })

    # CSV 저장
    csv_path = Path("/workspace/CCTV차종분류_output/final_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=final_labels[0].keys())
        w.writeheader()
        w.writerows(final_labels)
    logger.info("CSV: %s", csv_path)

    # 통계
    counts = {}
    for r in final_labels:
        counts[r["label_13cls"]] = counts.get(r["label_13cls"], 0) + 1

    logger.info("=" * 50)
    logger.info("최종 13종 라벨 분포:")
    train_total = 0
    for k in sorted(counts.keys()):
        v = counts[k]
        name = CLASS_NAMES.get(k, k)
        logger.info("  %-5s %s: %d장 (%.1f%%)", k, name, v, v / n * 100)
        if k not in ("NOISE", "MIXED"):
            train_total += v
    logger.info("  학습 가능 총: %d장", train_total)

    # verified_13cls 디렉토리 구조 생성 (심링크로)
    verified = Path("/workspace/CCTV차종분류_output/labels/verified_13cls")
    verified.mkdir(parents=True, exist_ok=True)

    logger.info("verified_13cls/ 디렉토리 생성 (심링크)...")
    for label in set(counts.keys()):
        name = CLASS_NAMES.get(label, label)
        (verified / name).mkdir(exist_ok=True)

    for r in tqdm(final_labels, desc="symlink"):
        src = Path(r["path"])
        name = r["label_name"]
        dst = verified / name / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except FileExistsError:
                pass

    logger.info("완료: %s", verified)


if __name__ == "__main__":
    main()
