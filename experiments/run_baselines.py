"""Run baseline methods.

基线对比实验（proposal 3.6 Baseline Comparison）：
4 个智能体（standard / min_cost / hard_stop / proposed）x 3 个预算档位
x 全部测试任务，指标写入 results/baselines.csv 与 baselines_summary.csv。

运行方式（项目根目录下）：
    python experiments/run_baselines.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import AGENT_NAMES, Evaluator, load_tasks

RESULTS_DIR = Path(__file__).resolve().parent / "results"


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
    rows = ev.run_suite(AGENT_NAMES, tasks, tiers, seed_base=seed_base)
    _write_csv(RESULTS_DIR / "baselines.csv", [r.to_dict() for r in rows])

    summary = []
    for agent in AGENT_NAMES:
        for tier in tiers:
            a = Evaluator.aggregate([
                r for r in rows
                if r.agent == agent and r.budget_tier == tier
            ])
            summary.append({"agent": agent, "budget_tier": tier, **a})
    _write_csv(RESULTS_DIR / "baselines_summary.csv", summary)

    print(f"\n{'agent':<12}{'tier':<8}{'succ%':>7}{'avg$':>8}{'judge':>7}")
    for s in summary:
        print(f"{s['agent']:<12}{s['budget_tier']:<8}"
              f"{s['success_rate'] * 100:>6.1f}{s['avg_spent']:>8.3f}"
              f"{s['avg_judge_overall']:>7.2f}")


if __name__ == "__main__":
    run()
