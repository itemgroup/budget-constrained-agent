"""Entry point: run single task / batch experiments.

用法（项目根目录下）：
    python main.py run --task "调研文档并生成摘要报告" --budget 1.0 [--seed 0]
    python main.py experiment baselines|ablation|budget_sweep|failure_test|all
"""

import argparse

from src.agent_core import AgentOrchestrator, RunReport
from src.simulated_services import SimulatedServiceRunner


def _print_report(report: RunReport) -> None:
    """打印单次任务运行报告。"""
    print(f"状态: {report.state.value} | 终止原因: {report.termination_reason}")
    print("执行日志:")
    for s in report.steps:
        if s.executed:
            outcome = "成功" if s.success else "失败"
            print(f"  轮次{s.step}: {s.subtask_desc} [{s.task_type}] -> "
                  f"{s.service_name} ({s.selection_status}) {outcome} "
                  f"cost=${s.cost:.2f} quality={s.quality:.2f}")
        else:
            print(f"  轮次{s.step}: {s.subtask_desc} [{s.task_type}] -> 未执行")
        print(f"    备注: {s.note}")
    if report.skipped_subtasks:
        print(f"被跳过（缺损）: {report.skipped_subtasks}")
    print(report.ledger_summary)


def cmd_run(args: argparse.Namespace) -> None:
    """运行单个任务。"""
    agent = AgentOrchestrator(
        runner=SimulatedServiceRunner(seed=args.seed),
        max_steps=args.max_steps,
    )
    _print_report(agent.run(args.task, args.budget))


def cmd_experiment(args: argparse.Namespace) -> None:
    """运行批量实验脚本。"""
    from experiments import (
        run_ablation,
        run_baselines,
        run_budget_sweep,
        run_failure_test,
    )

    runners = {
        "baselines": run_baselines.run,
        "ablation": run_ablation.run,
        "budget_sweep": run_budget_sweep.run,
        "failure_test": run_failure_test.run,
    }
    names = list(runners) if args.name == "all" else [args.name]
    for name in names:
        print("=" * 64)
        print(f"实验: {name}")
        print("=" * 64)
        runners[name](seed_base=args.seed_base)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="budget_agent",
        description="预算约束下的企业任务自主智能体（模拟环境）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="运行单个任务")
    p_run.add_argument("--task", required=True, help="自然语言任务描述")
    p_run.add_argument("--budget", type=float, default=1.0, help="总预算（美元）")
    p_run.add_argument("--seed", type=int, default=0, help="随机种子（可复现）")
    p_run.add_argument("--max-steps", type=int, default=20, help="最大轮次上限")

    p_exp = sub.add_parser("experiment", help="运行批量实验")
    p_exp.add_argument("name", choices=[
        "baselines", "ablation", "budget_sweep", "failure_test", "all",
    ])
    p_exp.add_argument("--seed-base", type=int, default=1000)

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        cmd_experiment(args)


if __name__ == "__main__":
    main()
