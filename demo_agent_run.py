"""Demo：AgentOrchestrator 核心调度器端到端运行。

案例1：预算充足，顺利跑完所有子任务（COMPLETED）。
案例2：子任务调用失败 -> 触发重规划 RETRY -> 重试成功 ->
       后续子任务预算不足 -> 预算耗尽终止（BUDGET_EXHAUSTED）。

运行方式（项目根目录下）：
    python demo_agent_run.py
"""

from src.agent_core import AgentOrchestrator, RunReport
from src.simulated_services import DEFAULT_FAILURE_RATES, SimulatedServiceRunner

TASK = "调研文档，分析要点并生成摘要报告"


def show_report(report: RunReport) -> None:
    """打印完整运行报告。"""
    print(f"最终状态: {report.state.value}")
    print(f"终止原因: {report.termination_reason}")
    print("-" * 64)
    print("执行日志:")
    for s in report.steps:
        if s.executed:
            outcome = "成功" if s.success else "失败"
            print(
                f"  轮次{s.step}: {s.subtask_desc} [{s.task_type}] "
                f"-> {s.service_name} ({s.selection_status}) "
                f"{outcome} cost=${s.cost:.2f} quality={s.quality:.2f}"
            )
        else:
            print(f"  轮次{s.step}: {s.subtask_desc} [{s.task_type}] -> 未执行")
        print(f"    备注: {s.note}")
    print("-" * 64)
    print(f"成功子任务结果（{len(report.outputs)} 个）:")
    for o in report.outputs:
        print(f"  [{o.task_type}] {o.subtask_desc} <- {o.service_name} "
              f"(quality {o.quality:.2f})")
        print(f"    输出: {o.output}")
    if report.skipped_subtasks:
        print(f"被跳过的子任务（任务缺损）: {report.skipped_subtasks}")
    print("-" * 64)
    print(report.ledger_summary)


def main() -> None:
    # ===================== 案例1：预算充足 =====================
    print("=" * 64)
    print("案例1：预算充足（$2.00），顺利跑完所有子任务")
    print("=" * 64)
    agent1 = AgentOrchestrator(runner=SimulatedServiceRunner(seed=0))
    show_report(agent1.run(TASK, total_budget=2.00))

    # ===================== 案例2：失败重试后预算耗尽 =====================
    print()
    print("=" * 64)
    print("案例2：预算紧张（$0.25）+ 注入 small_llm 50% 失败率")
    print("预期：LLM 子任务失败 -> RETRY 重试成功 -> 报告子任务预算不足 -> 终止")
    print("=" * 64)
    # 失败注入：仅 small_llm 有 50% 失败率，其余服务置 0（演示确定性）
    rates = {name: 0.0 for name in DEFAULT_FAILURE_RATES}
    rates["small_llm"] = 0.5
    agent2 = AgentOrchestrator(runner=SimulatedServiceRunner(failure_rates=rates, seed=3))
    show_report(agent2.run(TASK, total_budget=0.25))


if __name__ == "__main__":
    main()
