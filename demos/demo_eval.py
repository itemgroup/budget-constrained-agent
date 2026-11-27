"""Demo：评估模块端到端验证。

对 20 个企业测试任务 x 4 个智能体（3 基线 + 完整版）x 3 个预算档位
批量运行（共 240 次模拟），按 智能体 x 档位 汇总指标并打印对比表；
另附消融变体（full / 禁预算感知选择 / 禁动态重规划）汇总。

运行方式（项目根目录下）：
    python demos/demo_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import (
    AGENT_NAMES,
    ABLATION_NAMES,
    Evaluator,
    load_tasks,
)


def print_table(title: str, names: tuple[str, ...], rows, tiers: dict) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)
    header = (
        f"{'agent':<26}{'tier':<8}{'succ%':>7}{'avg$':>8}{'over%':>7}"
        f"{'calls':>7}{'repeat':>8}{'compl':>7}{'corr':>7}{'judge':>7}"
    )
    print(header)
    print("-" * 100)
    ev = Evaluator()
    for agent_name in names:
        for tier_name in tiers:
            sub = [
                r for r in rows
                if r.agent == agent_name and r.budget_tier == tier_name
            ]
            a = ev.aggregate(sub)
            print(
                f"{agent_name:<26}{tier_name:<8}"
                f"{a['success_rate'] * 100:>6.1f}"
                f"{a['avg_spent']:>8.3f}"
                f"{a['over_budget_ratio'] * 100:>6.1f}"
                f"{a['avg_tool_calls']:>7.1f}"
                f"{a['avg_repeated_calls']:>8.2f}"
                f"{a['avg_completeness']:>7.2f}"
                f"{a['avg_correctness']:>7.2f}"
                f"{a['avg_judge_overall']:>7.2f}"
            )
        print("-" * 100)


def main() -> None:
    ev = Evaluator()
    tasks, tiers = load_tasks()
    print(f"任务数: {len(tasks)}，预算档位: {tiers}")

    # 基线对比（proposal 3.4 / 3.6 Baseline Comparison）
    rows = ev.run_suite(AGENT_NAMES, tasks, tiers)
    print_table("基线对比（4 智能体 x 3 档位 x 20 任务）", AGENT_NAMES, rows, tiers)

    # 消融实验（proposal 3.6 Ablation Study）
    ab_rows = ev.run_suite(ABLATION_NAMES, tasks, tiers, ablation=True)
    print_table("消融实验（完整版 vs 禁用单模块）", ABLATION_NAMES, ab_rows, tiers)


if __name__ == "__main__":
    main()
