"""Run ablation experiments.

消融实验（proposal 3.6 Ablation Study）：
完整版 vs 禁用预算感知选择 vs 禁用动态重规划，
3 个变体 x 3 个预算档位 x 全部测试任务，
指标写入 results/ablation.csv 与 ablation_summary.csv。

运行方式（项目根目录下）：
    python experiments/run_ablation.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import ABLATION_NAMES, Evaluator, load_tasks

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
    rows = ev.run_suite(ABLATION_NAMES, tasks, tiers,
                        seed_base=seed_base, ablation=True)
    _write_csv(RESULTS_DIR / "ablation.csv", [r.to_dict() for r in rows])

    summary = []
    for variant in ABLATION_NAMES:
        for tier in tiers:
            a = Evaluator.aggregate([
                r for r in rows
                if r.agent == variant and r.budget_tier == tier
            ])
            summary.append({"variant": variant, "budget_tier": tier, **a})
    _write_csv(RESULTS_DIR / "ablation_summary.csv", summary)

    print(f"\n{'variant':<26}{'tier':<8}{'succ%':>7}{'avg$':>8}{'judge':>7}")
    for s in summary:
        print(f"{s['variant']:<26}{s['budget_tier']:<8}"
              f"{s['success_rate'] * 100:>6.1f}{s['avg_spent']:>8.3f}"
              f"{s['avg_judge_overall']:>7.2f}")


if __name__ == "__main__":
    run()
