"""High/Medium/Low budget comparison.

多预算对比实验（proposal 3.6 Multi-Budget Comparison）：
完整版智能体在 high / medium / low 三档预算下运行同一测试集，
观察资源约束如何改变服务选择模式与整体任务表现。

输出：results/budget_sweep.csv、budget_sweep_summary.csv、
     budget_sweep_service_usage.csv（各档位下的服务使用分布）。

运行方式（项目根目录下）：
    python experiments/run_budget_sweep.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import Evaluator, build_agent, load_tasks

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
    rows = []
    usage: dict[str, Counter] = {}
    for tier_name, tier_budget in tiers.items():
        usage[tier_name] = Counter()
        for i, t in enumerate(tasks):
            agent = build_agent("proposed", seed=seed_base + i)
            rows.append(ev.run_single(agent, "proposed", t,
                                      tier_name, tier_budget))
            # run_single 执行后，智能体实例仍保留本次执行上下文
            for s in agent.steps:
                if s.executed:
                    usage[tier_name][s.service_name] += 1
    _write_csv(RESULTS_DIR / "budget_sweep.csv", [r.to_dict() for r in rows])

    summary = []
    for tier in tiers:
        a = Evaluator.aggregate([r for r in rows if r.budget_tier == tier])
        summary.append({"budget_tier": tier, **a})
    _write_csv(RESULTS_DIR / "budget_sweep_summary.csv", summary)

    usage_rows = [
        {"budget_tier": tier, "service": svc, "calls": cnt}
        for tier, counter in usage.items()
        for svc, cnt in counter.most_common()
    ]
    _write_csv(RESULTS_DIR / "budget_sweep_service_usage.csv", usage_rows)

    print(f"\n{'tier':<8}{'succ%':>7}{'avg$':>8}{'calls':>7}"
          f"{'repeat':>8}{'judge':>7}")
    for s in summary:
        print(f"{s['budget_tier']:<8}{s['success_rate'] * 100:>6.1f}"
              f"{s['avg_spent']:>8.3f}{s['avg_tool_calls']:>7.1f}"
              f"{s['avg_repeated_calls']:>8.2f}{s['avg_judge_overall']:>7.2f}")
    print("\n各档位服务选择分布（调用次数）:")
    for tier, counter in usage.items():
        dist = ", ".join(f"{svc}x{cnt}" for svc, cnt in counter.most_common())
        print(f"  {tier:<8}: {dist}")


if __name__ == "__main__":
    run()
