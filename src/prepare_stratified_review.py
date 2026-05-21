"""클래스별 고신뢰 상위 N장 층화 추출 → Parquet is_active_sample 마킹."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PARQUET = Path("/workspace/prj_cctv/pipeline/data/labeling_v1/crops.parquet")

TARGET_CLASSES = {
    "T2": 100,
    "T3": 100,
    "T4": 100,
    "T5": 100,
    "T10": 200,  # 전수 (198장)
}


def run(parquet_path: Path = PARQUET, targets: dict | None = None, daytime_only: bool = False):
    targets = targets or TARGET_CLASSES

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"Parquet 로드: {len(df)} rows")

    df["is_active_sample"] = False

    selected_ids = []
    for cls, n in targets.items():
        mask = df["current_class"] == cls
        if daytime_only:
            mask &= df["video_category"].isin(["day", "backlight"])
        pool = df[mask].sort_values("current_conf", ascending=False)
        chosen = pool.head(n)
        selected_ids.extend(chosen["crop_id"].tolist())

        actual = len(chosen)
        conf_min = chosen["current_conf"].min() if actual > 0 else 0
        conf_max = chosen["current_conf"].max() if actual > 0 else 0
        vc = chosen["video_category"].value_counts().to_dict() if actual > 0 else {}
        print(f"  {cls}: {actual}/{n}장 선택, conf [{conf_min:.3f} ~ {conf_max:.3f}], 조건={vc}")

    df.loc[df["crop_id"].isin(selected_ids), "is_active_sample"] = True

    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(new_table, parquet_path, compression="zstd", compression_level=3)

    total = len(selected_ids)
    print(f"\n총 {total}장 마킹 완료 → is_active_sample=True")
    print(f"GUI 실행: python3 crop_review_gui.py --max {total}")

    summary = {
        "total_selected": total,
        "targets": targets,
        "daytime_only": daytime_only,
        "per_class": {},
    }
    for cls in targets:
        cnt = df[(df["crop_id"].isin(selected_ids)) & (df["current_class"] == cls)].shape[0]
        summary["per_class"][cls] = cnt

    summary_path = parquet_path.parent / "stratified_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"요약: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(PARQUET))
    parser.add_argument("--daytime-only", action="store_true")
    args = parser.parse_args()
    run(Path(args.parquet), daytime_only=args.daytime_only)
