"""Hybrid 검증: 원본 6종(YOLO) vs 13종(클러스터) 라벨 비교

검증 항목:
  1. 6종 ↔ 13종 매핑 일치도 (예상되는 매핑)
  2. 클래스별 불일치율
  3. 의심 케이스 추출 (불일치 + 노이즈가 아닌 것)
  4. 클러스터 응집도 (한 클러스터의 원본 6종 분포)

산출:
  - verification_report.txt
  - confusion_matrix.csv
  - suspicious_cases.csv (수동 검토 필요)
"""

import csv
import logging
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# 6종 → 13종 예상 매핑
EXPECTED_MAPPING = {
    "C1_승용차":   ["T1"],          # 1종
    "C2_버스":     ["T2"],          # 2종
    "C3_소형화물": ["T3"],          # 3종
    "C4_중형화물": ["T4"],          # 4종
    "C5_대형화물": ["T5", "T6", "T8", "T10"],  # 5/6/8/10종
    "C6_이륜차":   ["T13"],         # 13종
}


def main():
    cluster_path = Path("/workspace/CCTV차종분류_output/clusters/cluster_assignments.npz")
    final_csv = Path("/workspace/CCTV차종분류_output/final_labels.csv")

    # 원본 6종 로드 (cluster_assignments에 있음)
    cluster_data = np.load(cluster_path, allow_pickle=True)
    paths = cluster_data["paths"]
    orig_labels = cluster_data["labels"]
    n = len(paths)

    path_to_orig = {p: l for p, l in zip(paths, orig_labels)}

    # 13종 라벨 로드
    rows = []
    with open(final_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["original_6cls"] = path_to_orig.get(r["path"], "")
            rows.append(r)

    logger.info("로드: %d행", len(rows))

    # ========== 1. Confusion Matrix ==========
    confusion = defaultdict(lambda: Counter())
    for r in rows:
        orig = r["original_6cls"]
        new = r["label_13cls"]
        confusion[orig][new] += 1

    logger.info("=" * 70)
    logger.info("Confusion Matrix (원본 6종 → 13종)")
    logger.info("=" * 70)

    all_new_labels = sorted(set(new for c in confusion.values() for new in c.keys()))
    header = f"{'원본':15s} | " + " | ".join(f"{l:6s}" for l in all_new_labels) + " | 합계"
    logger.info(header)
    logger.info("-" * len(header))

    for orig in sorted(confusion.keys()):
        counts = confusion[orig]
        total = sum(counts.values())
        line = f"{orig:15s} | " + " | ".join(f"{counts.get(l, 0):6d}" for l in all_new_labels) + f" | {total:6d}"
        logger.info(line)

    # ========== 2. 일치율 분석 ==========
    logger.info("=" * 70)
    logger.info("매핑 일치율 (예상 매핑 대비)")
    logger.info("=" * 70)

    total_match = 0
    total_check = 0
    for orig, expected in EXPECTED_MAPPING.items():
        counts = confusion.get(orig, Counter())
        total_orig = sum(counts.values())
        if total_orig == 0:
            continue
        match = sum(counts[e] for e in expected)
        # 노이즈/혼합 제외하고 계산
        valid = total_orig - counts.get("NOISE", 0) - counts.get("MIXED", 0)
        match_rate = match / valid * 100 if valid > 0 else 0

        # 가장 많이 흘러간 잘못된 라벨
        wrong = {k: v for k, v in counts.items() if k not in expected and k not in ("NOISE", "MIXED")}
        top_wrong = sorted(wrong.items(), key=lambda x: -x[1])[:3]
        wrong_str = " ".join(f"{k}:{v}" for k, v in top_wrong)

        logger.info("  %-15s 예상=%s | 일치 %d/%d (%.1f%%) | 오류 %s",
                    orig, expected, match, valid, match_rate, wrong_str or "-")
        total_match += match
        total_check += valid

    overall = total_match / total_check * 100 if total_check > 0 else 0
    logger.info("  ── 전체 일치율: %d/%d (%.1f%%)", total_match, total_check, overall)

    # ========== 3. 의심 케이스 추출 ==========
    logger.info("=" * 70)
    logger.info("의심 케이스 (예상 매핑과 불일치, 노이즈/혼합 제외)")
    logger.info("=" * 70)

    suspicious = []
    for r in rows:
        orig = r["original_6cls"]
        new = r["label_13cls"]
        if new in ("NOISE", "MIXED"):
            continue
        expected = EXPECTED_MAPPING.get(orig, [])
        if expected and new not in expected:
            suspicious.append(r)

    logger.info("총 의심 케이스: %d장 (%.1f%%)", len(suspicious), len(suspicious) / n * 100)

    # 의심 케이스를 6종→13종 패턴별로 카운트
    pattern_counts = Counter()
    for r in suspicious:
        pattern_counts[(r["original_6cls"], r["label_13cls"])] += 1

    logger.info("의심 패턴 (상위 10):")
    for (orig, new), cnt in pattern_counts.most_common(10):
        logger.info("  %s → %s: %d장", orig, new, cnt)

    # 저장
    susp_csv = Path("/workspace/CCTV차종분류_output/suspicious_cases.csv")
    with open(susp_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["path", "original_6cls", "cluster_id", "label_13cls", "label_name", "source"])
        w.writeheader()
        w.writerows(suspicious)
    logger.info("저장: %s", susp_csv)

    # ========== 4. 클러스터 응집도 ==========
    logger.info("=" * 70)
    logger.info("클러스터 응집도 (낮은 순 = 잡탕 클러스터)")
    logger.info("=" * 70)

    cluster_purity = []
    for r in rows:
        cid = int(r["cluster_id"])
        cluster_purity.append((cid, r["original_6cls"]))

    cluster_dist = defaultdict(Counter)
    for cid, orig in cluster_purity:
        cluster_dist[cid][orig] += 1

    purities = []
    for cid in sorted(cluster_dist.keys()):
        dist = cluster_dist[cid]
        total = sum(dist.values())
        max_cls = dist.most_common(1)[0]
        purity = max_cls[1] / total
        purities.append((cid, total, purity, max_cls[0]))

    # 응집도 낮은 10개
    purities.sort(key=lambda x: x[2])
    logger.info("응집도 낮은 10개:")
    for cid, total, purity, top_cls in purities[:10]:
        logger.info("  C%02d (n=%d): %.0f%% %s", cid, total, purity * 100, top_cls)

    logger.info("응집도 높은 10개:")
    for cid, total, purity, top_cls in purities[-10:]:
        logger.info("  C%02d (n=%d): %.0f%% %s", cid, total, purity * 100, top_cls)


if __name__ == "__main__":
    main()
