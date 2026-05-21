"""학습 라운드 중단 조건 (7개) 체크.

Round 3 자동 재라벨링 실패(GT -1.7%p)의 재발을 방지.
매 라운드 종료 시 호출; 하나라도 trip되면 루프 중단 + 롤백.

조건:
1. 홀드아웃 정확도 하락 -2.0%p
2. Conf 인플레이션 +5.0%p (통과율 증가 but 정확도 정체)
3. 클래스 소실 -30% 또는 50장 미만
4. 정체 ±1.0%p 2라운드 연속
5. 최대 라운드 (5회 hard cap, 운영 상한 3)
6. val-GT 갭 역전 (과적합 재발)
7. Claude 검수 누수율 >10% (pool 위반)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import config

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """라운드 종료 시 기록."""
    round_num: int
    timestamp: str
    holdout_acc: float
    val_acc: float
    per_class_acc: dict[str, float]   # {"T1": 0.85, ...}
    per_class_n:   dict[str, int]     # training sample counts
    pass_rate:     float              # conf >= 0.8 비율
    pool_leak_rate: float = 0.0
    notes: str = ""


@dataclass
class StopDecision:
    """중단 판정 결과."""
    triggered: bool
    condition: str         # "" if not triggered
    detail: str
    recommendation: str    # "rollback" | "halt" | "continue"


def check_stop(
    current: RoundMetrics,
    previous: list[RoundMetrics],
) -> StopDecision:
    """현재 라운드가 중단 조건에 해당하는지."""
    # #5 Max rounds
    if current.round_num > config.MAX_ROUNDS:
        return StopDecision(True, "MAX_ROUNDS",
                            f"round {current.round_num} > {config.MAX_ROUNDS}",
                            "halt")
    if current.round_num > config.ROUND_HARD_CAP:
        logger.warning("round %d exceeds operational cap %d",
                       current.round_num, config.ROUND_HARD_CAP)

    # #7 Review leak
    if current.pool_leak_rate > config.STOP_REVIEW_LEAK:
        return StopDecision(True, "REVIEW_LEAK",
                            f"leak={current.pool_leak_rate*100:.1f}% > {config.STOP_REVIEW_LEAK*100:.1f}%",
                            "halt")

    if not previous:
        return StopDecision(False, "", "first round — baseline only", "continue")

    prev = previous[-1]
    delta_holdout = current.holdout_acc - prev.holdout_acc

    # #1 Holdout drop
    if delta_holdout <= config.STOP_DROP_HOLDOUT:
        return StopDecision(True, "HOLDOUT_DROP",
                            f"Δ={delta_holdout*100:+.2f}%p (≤ {config.STOP_DROP_HOLDOUT*100}%p)",
                            "rollback")

    # #6 val-GT gap reversal (과적합 재발)
    # val >> holdout이 커지는 경우만 위험 — 반대 방향(holdout>val)은 단순히
    # val 샘플이 어려웠던 noise로 해석.
    current_gap = current.val_acc - current.holdout_acc
    prev_gap    = prev.val_acc - prev.holdout_acc
    if (current_gap > 0 and
        current_gap > prev_gap + config.STOP_GAP_REVERSAL):
        return StopDecision(True, "GAP_REVERSAL",
                            f"val-GT gap {prev_gap*100:.1f}%p → {current_gap*100:.1f}%p "
                            f"(+{(current_gap-prev_gap)*100:.1f}%p)",
                            "rollback")

    # #2 Conf inflation (pass_rate↑ but acc not↑)
    delta_pass = current.pass_rate - prev.pass_rate
    if delta_pass >= config.STOP_CONF_INFLATE and delta_holdout < 0.01:
        return StopDecision(True, "CONF_INFLATION",
                            f"pass_rate {prev.pass_rate:.2%}→{current.pass_rate:.2%} "
                            f"but holdout Δ={delta_holdout*100:+.2f}%p",
                            "halt")

    # #3 Class loss
    for cls in config.CLASS_ORDER:
        p = prev.per_class_n.get(cls, 0)
        c = current.per_class_n.get(cls, 0)
        if p > 0 and (c / p - 1) <= config.STOP_CLASS_LOSS_PCT:
            return StopDecision(True, "CLASS_LOSS",
                                f"{cls}: {p} → {c} ({(c/p-1)*100:+.0f}%)",
                                "halt")
        if c < config.STOP_CLASS_LOSS_ABS:
            return StopDecision(True, "CLASS_LOSS",
                                f"{cls}: {c} < {config.STOP_CLASS_LOSS_ABS}",
                                "halt")

    # #4 Stagnation (2 rounds consecutive within ±1%p)
    if len(previous) >= 1:
        if abs(delta_holdout) < config.STOP_STAGNATION_EPS:
            prev2_delta = (prev.holdout_acc - previous[-2].holdout_acc) if len(previous) >= 2 else 1.0
            if abs(prev2_delta) < config.STOP_STAGNATION_EPS:
                return StopDecision(True, "STAGNATION",
                                    f"2 rounds within ±{config.STOP_STAGNATION_EPS*100}%p",
                                    "halt")

    return StopDecision(False, "", f"Δholdout={delta_holdout*100:+.2f}%p — continue", "continue")


def log_round(metrics: RoundMetrics, log_path: Path) -> None:
    """Append-only refinement_log.jsonl."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")


def load_history(log_path: Path) -> list[RoundMetrics]:
    if not log_path.exists():
        return []
    return [RoundMetrics(**json.loads(l)) for l in log_path.read_text().splitlines()]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check")
    p.add_argument("--log", default=str(config.ROUNDS_DIR / "refinement_log.jsonl"))
    p.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    hist = load_history(Path(args.log))
    if not hist:
        print("no history")
        raise SystemExit(0)
    current = hist[-1]
    previous = hist[:-1]
    decision = check_stop(current, previous)
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))
    raise SystemExit(1 if decision.triggered else 0)
