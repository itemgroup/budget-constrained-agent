"""Run failure-handling tests.

工具失败注入实验（proposal 3.6 Tool-Failure Test）：
对全部服务按倍率提升失败率（1x 基准 / 5x / 10x），观察智能体是否
切换替代服务，并量化失败处理带来的额外开销（调用次数、重复调用、开销）。

输出：results/failure_test.csv 与 failure_test_summary.csv。

运行方式（项目根目录下）：
    python experiments/run_failure_test.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import Evaluator, load_tasks
from src.multi_agent.orchestrator import build_enterprise_multi_agent
from src.simulated_services import DEFAULT_FAILURE_RATES, SimulatedServiceRunner

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 失败注入倍率：1x = 配置基准失败率
FAILURE_LEVELS = {"normal_1x": 1.0, "elevated_5x": 5.0, "severe_10x": 10.0}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存: {path}")


def run(seed_base: int = 1000) -> None:
    ev = Evaluator()
    tasks, tiers = load_tasks()
    rows, levels = [], []
    for level_name, factor in FAILURE_LEVELS.items():
        rates = {
            name: min(0.95, p * factor)
            for name, p in DEFAULT_FAILURE_RATES.items()
        }
        for tier_name, tier_budget in tiers.items():
            for i, t in enumerate(tasks):
                agent = build_enterprise_multi_agent(
                    seed=seed_base + i,
                    runner=SimulatedServiceRunner(
                        failure_rates=rates, seed=seed_base + i,
                    ),
                )
                rows.append(ev.run_single(agent, "proposed", t,
                                          tier_name, tier_budget))
                levels.append(level_name)

    _write_csv(RESULTS_DIR / "failure_test.csv", [
        {"failure_level": lv, **m.to_dict()} for lv, m in zip(levels, rows)
    ])

    summary = []
    for level_name in FAILURE_LEVELS:
        for tier in tiers:
            sub = [m for lv, m in zip(levels, rows)
                   if lv == level_name and m.budget_tier == tier]
            a = Evaluator.aggregate(sub)
            summary.append({"failure_level": level_name,
                            "budget_tier": tier, **a})
    _write_csv(RESULTS_DIR / "failure_test_summary.csv", summary)

    print(f"\n{'level':<13}{'tier':<8}{'succ%':>7}{'avg$':>8}"
          f"{'calls':>7}{'repeat':>8}{'judge':>7}")
    for s in summary:
        print(f"{s['failure_level']:<13}{s['budget_tier']:<8}"
              f"{s['success_rate'] * 100:>6.1f}{s['avg_spent']:>8.3f}"
              f"{s['avg_tool_calls']:>7.1f}{s['avg_repeated_calls']:>8.2f}"
              f"{s['avg_judge_overall']:>7.2f}")


if __name__ == "__main__":
    run()
