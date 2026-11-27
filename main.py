"""Entry point: run single task / batch experiments.

用法（项目根目录下）：
    python main.py run --task "调研文档并生成摘要报告" --budget 1.0 [--seed 0]
    python main.py experiment baselines|ablation|budget_sweep|failure_test|all
"""

import argparse

from src.agent_core import RunReport


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
    planner = None
    judge = None
    if args.llm:
        # 真实 LLM：任务分解 + 结果评分；无 DEEPSEEK_API_KEY 时各自回退模拟实现
        from src.evaluation.llm_judge import RealLLMJudge
        from src.llm_task_planner import LLMTaskPlanner

        rag = None
        if args.rag:
            from src.rag import RAGRetriever

            rag = RAGRetriever()
            info = rag.stats()
            print(f"[RAG] 后端={info['backend']} 记录={info['records']} "
                  f"（已过滤 mock {info['mock_skipped']} 条）")

        planner = LLMTaskPlanner(rag=rag)
        judge = RealLLMJudge()
    # 多 Agent 编排（默认且唯一形态）：调研 -> 询价 -> 分解 -> 主循环 -> 审计
    from src.multi_agent.orchestrator import build_enterprise_multi_agent

    multi_agent = build_enterprise_multi_agent(
        seed=args.seed, planner=planner, judge=judge,
        max_steps=args.max_steps,
    )
    report = multi_agent.run(args.task, args.budget)
    _print_report(report)
    if planner is not None:
        if planner.llm_success:
            source = "真实 LLM"
        elif planner.llm_calls:
            source = "模拟（LLM 调用失败回退）"
        else:
            source = "模拟"
        print(f"分解来源: {source}（LLM 调用 {planner.llm_calls} 次，"
              f"记忆命中 {planner.memory_hits} 次）")
        hits = getattr(planner, "last_rag_hits", [])
        if hits:
            print(f"历史任务参考: 匹配 {len(hits)} 条"
                  f"（最高相似度 {hits[0].score:.2f}）")
    audit = report.audit_report
    if audit is not None:
        print(f"审计: {'通过' if audit.passed else '驳回'} | "
              f"对账 {'一致' if audit.reconciliation_ok else '异常'} | "
              f"格式问题 {len(audit.format_errors)} 项")
        if audit.score is not None:
            print(f"Judge: completeness={audit.score.completeness:.2f} "
                  f"correctness={audit.score.correctness:.2f} "
                  f"overall={audit.score.overall:.2f}")
        print(f"审计说明: {audit.reason}")
    roles = "、".join(sorted({m.agent for m in report.messages}))
    print(f"协作日志 {len(report.messages)} 条 | 参与 agent: {roles}")
    try:
        from src.run_log import append_run, record_from_report

        real_llm = bool(planner is not None
                        and getattr(planner, "llm_success", False))
        append_run(record_from_report(
            args.task, args.budget, report, None, report.audit_report,
            mode="llm" if real_llm else "mock",
        ))
    except Exception:  # noqa: BLE001 - 记录失败不影响运行
        pass


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
    p_run.add_argument("--llm", action="store_true",
                       help="用真实 LLM（OpenAI 兼容端点）做任务分解与结果评分")
    p_run.add_argument("--rag", action="store_true",
                       help="分解前参考历史任务记录（需配合 --llm）")

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
