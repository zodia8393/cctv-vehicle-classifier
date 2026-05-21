"""Claude 세션 검증 워크플로우 (API 불사용).

gt_builder.py가 생성한 4×4 그리드를 이 Claude Code 세션에서 Read 도구로 읽고
응답을 JSONL에 기록하는 프로토콜. API 키 불필요.

워크플로우:
  1. gt_builder.py grid ... → grids/*.png + grids/*.json
  2. claude_review.py init-queue → responses/ 디렉토리에 빈 템플릿 생성
  3. (세션에서) 각 grid_NNN.png을 Read 후 grid_NNN_response.json에 cell별 라벨 기입
  4. claude_review.py merge → manifest에 claude_class 반영
  5. (사용자) 수동 확정 (disagreements 위주) → final_class 기입
  6. claude_review.py finalize → manifest.jsonl + manifest.sha256 봉인

중단 조건 #7 (Claude 검수 누수율): merge 시 pool 위반 crop 자동 거부.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import config
from gt_builder import (
    CropRecord, load_partition, seal_manifest, write_manifest, verify_manifest,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── 프롬프트 템플릿 (Claude 세션에서 참고) ──────────────────────────

REVIEW_PROMPT = f"""당신은 한국 고속도로 CCTV 영상의 차량 차종 분류 전문가입니다.
4×4 그리드 이미지를 읽고 각 셀(왼쪽 상단부터 1~16)의 차량을 아래 7종 중 하나로 분류하세요.

| 코드 | 명칭 | 특징 |
|------|------|------|
| T1 | 승용차 | 일반 승용차 (sedan, SUV, 해치백) |
| T2 | 버스 | 고속/관광/시내버스, 4m+ 높이 |
| T3 | 소형화물 | 1톤 트럭 (포터, 봉고), 짐칸 있음 |
| T4 | 중형화물 | 2.5~5톤 트럭, 박스 트럭 포함 |
| T5 | 대형화물 | 5톤 이상, 덤프/믹서/카고 |
| T10 | 세미트레일러 | 트랙터+트레일러, 5축 이상 |
| T13 | 이륜차 | 오토바이, 스쿠터 |

분류 불가능한 경우 "UNK" 기입.

응답 형식 (JSON):
{{
  "grid_id": "grid_NNN",
  "cells": [
    {{"cell": 1, "class": "T1", "confidence": "high"}},
    {{"cell": 2, "class": "T4", "confidence": "medium"}},
    ...
  ],
  "notes": ""
}}

confidence: "high" (확실) | "medium" (가능성 높음) | "low" (불확실) | "unk" (판단 불가)
"""


# ── 큐 초기화 / 병합 ───────────────────────────────────────────────

def init_queue(grids_dir: Path, out_dir: Path) -> None:
    """각 grid에 대응하는 빈 응답 템플릿 생성."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_jsons = sorted(grids_dir.glob("grid_*.json"))
    for gj in grid_jsons:
        meta = json.loads(gj.read_text())
        grid_id = gj.stem
        tpl = {
            "grid_id": grid_id,
            "grid_path": meta["grid_path"],
            "cells": [
                {"cell": c["cell"], "crop_path": c["crop_path"], "class": "UNK",
                 "confidence": "unk"}
                for c in meta["cells"]
            ],
            "notes": "",
        }
        out_path = out_dir / f"{grid_id}_response.json"
        if not out_path.exists():
            out_path.write_text(json.dumps(tpl, ensure_ascii=False, indent=2))
    logger.info("queue initialized: %d grid templates in %s", len(grid_jsons), out_dir)
    logger.info("prompt template (to guide in-session classification):\n%s", REVIEW_PROMPT)


def merge_responses(
    crop_manifest: Path, responses_dir: Path, out_manifest: Path,
    enforce_pool: str = "gt",
) -> dict[str, int]:
    """Claude 응답을 crop manifest에 병합 → claude_class 기입.

    pool 위반 crop은 즉시 거부 (stop condition #7).
    """
    partition = load_partition()
    records = [CropRecord(**json.loads(l)) for l in crop_manifest.read_text().splitlines()]
    # index by crop_path
    by_crop = {r.crop_path: r for r in records}

    stats = {"merged": 0, "pool_violation": 0, "unk": 0, "missing": 0}
    for resp_file in sorted(responses_dir.glob("grid_*_response.json")):
        resp = json.loads(resp_file.read_text())
        for cell in resp["cells"]:
            crop_path = cell["crop_path"]
            rec = by_crop.get(crop_path)
            if rec is None:
                stats["missing"] += 1
                continue
            if partition.get(rec.video_hash) != enforce_pool:
                stats["pool_violation"] += 1
                logger.error("POOL VIOLATION rejected: %s", crop_path)
                continue
            cls = cell["class"]
            if cls == "UNK":
                stats["unk"] += 1
                rec.claude_class = None
            elif cls in config.CLASS_ORDER:
                rec.claude_class = cls
                rec.notes = f"conf={cell.get('confidence', '?')}"
                stats["merged"] += 1
            else:
                logger.warning("unknown class '%s' in %s cell %d", cls, resp_file.name, cell["cell"])
    write_manifest(list(by_crop.values()), out_manifest)
    leak_rate = stats["pool_violation"] / max(len(records), 1)
    logger.info("merge: %s (leak_rate=%.2f%%)", stats, leak_rate * 100)
    if leak_rate > config.STOP_REVIEW_LEAK:
        raise RuntimeError(f"STOP CONDITION #7: review leak rate {leak_rate*100:.1f}% > {config.STOP_REVIEW_LEAK*100:.1f}%")
    return stats


def finalize(manifest: Path, require_final: bool = True) -> str:
    """수동 확정(final_class) 검증 후 SHA256 봉인."""
    records = [CropRecord(**json.loads(l)) for l in manifest.read_text().splitlines()]
    if require_final:
        missing = [r.crop_path for r in records if not r.final_class]
        if missing:
            raise RuntimeError(f"final_class missing for {len(missing)} records (first: {missing[0]})")
    return seal_manifest(manifest)


def compute_class_distribution(manifest: Path) -> dict[str, int]:
    records = [CropRecord(**json.loads(l)) for l in manifest.read_text().splitlines()]
    dist: dict[str, int] = {c: 0 for c in config.CLASS_ORDER}
    dist["UNK"] = 0
    for r in records:
        key = r.final_class or r.claude_class or "UNK"
        dist[key] = dist.get(key, 0) + 1
    return dist


def check_gt_targets(manifest: Path) -> tuple[bool, dict]:
    """GT 클래스별 목표치 충족 여부."""
    dist = compute_class_distribution(manifest)
    ok = True
    shortfall = {}
    for c in config.CLASS_ORDER:
        target = config.GT_T10 if c == "T10" else config.GT_PER_CLASS
        if dist.get(c, 0) < target:
            ok = False
            shortfall[c] = target - dist.get(c, 0)
    return ok, {"distribution": dist, "shortfall": shortfall}


# ── CLI ────────────────────────────────────────────────────────────

def cmd_init_queue(args):
    init_queue(Path(args.grids_dir), Path(args.out_dir))


def cmd_prompt(args):
    """현재 세션의 Claude에게 전달할 프롬프트 출력."""
    print(REVIEW_PROMPT)


def cmd_merge(args):
    merge_responses(
        Path(args.crop_manifest), Path(args.responses_dir), Path(args.out_manifest),
        enforce_pool=args.pool,
    )


def cmd_finalize(args):
    finalize(Path(args.manifest), require_final=not args.skip_final_check)


def cmd_status(args):
    ok, report = check_gt_targets(Path(args.manifest))
    status = "PASS" if ok else "SHORTFALL"
    print(f"[{status}] class distribution:")
    for c, n in report["distribution"].items():
        print(f"  {c}: {n}")
    if report["shortfall"]:
        print(f"shortfall: {report['shortfall']}")


def main():
    parser = argparse.ArgumentParser(description="Claude session-based review workflow")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-queue", help="빈 응답 템플릿 생성")
    p.add_argument("--grids-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_init_queue)

    p = sub.add_parser("prompt", help="세션 분류 프롬프트 출력")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("merge", help="응답 병합 → crop manifest에 claude_class 기입")
    p.add_argument("--crop-manifest", required=True)
    p.add_argument("--responses-dir", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--pool", default="gt", choices=["gt", "train"])
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("finalize", help="final_class 확인 → SHA256 봉인")
    p.add_argument("--manifest", required=True)
    p.add_argument("--skip-final-check", action="store_true")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("status", help="클래스 분포 + 목표 충족 상태")
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
