"""Demo：动态重规划模块的三种分支演示。

构造两类触发场景（执行失败 / 质量不足），覆盖 RETRY（同质量重试、
降级重试）、SKIP、ABORT 三种处理策略。

运行方式（项目根目录下）：
    python demos/demo_replan.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.replanner import Replanner
from src.service_registry import ServiceRegistry
from src.simulated_services import ServiceResult
from src.task_planner import Subtask


def make_subtask(order: int, task_type: str, min_quality: float, desc: str) -> Subtask:
    """构造演示用子任务。"""
    return Subtask(
        order=order,
        task_type=task_type,
        required_min_quality=min_quality,
        estimated_cost=0.0,
        desc=desc,
    )


def show_case(
    replanner: Replanner,
    title: str,
    current: Subtask,
    result: ServiceResult,
    remaining: list[Subtask],
    budget: float,
) -> None:
    """执行一次重规划并打印结果。"""
    print("=" * 64)
    print(title)
    print(f"  当前子任务: {current.desc} [{current.task_type}] "
          f"最低质量 {current.required_min_quality:.2f}")
    print(f"  执行结果: success={result.success}, quality={result.quality:.2f}"
          + (f", error={result.error_msg}" if result.error_msg else ""))
    print(f"  剩余预算: ${budget:.2f}，后续子任务: {[st.desc for st in remaining]}")

    # 先展示触发判断，再做重规划
    triggered = replanner.should_trigger(current, result)
    plan = replanner.replan(current, result, remaining, budget)
    print(f"  触发重规划: {triggered}")
    print(f"  -> 动作: {plan.action.value}")
    print(f"  -> 原因: {plan.reason}")
    print(f"  -> 更新后队列: {[st.desc for st in plan.updated_subtasks]}")
    print()


def main() -> None:
    registry = ServiceRegistry()
    replanner = Replanner(registry)

    search = make_subtask(1, "Search", 0.70, "检索相关资料")
    parse = make_subtask(2, "Document Parsing", 0.60, "解析文档内容")
    llm = make_subtask(3, "LLM Reasoning", 0.75, "分析并推理关键信息")
    report = make_subtask(4, "Report Generation", 0.65, "生成摘要报告")

    fail_result = ServiceResult(
        success=False, output="", cost=0.0, quality=0.0,
        error_msg="模拟随机失败",
    )
    low_quality_result = ServiceResult(
        success=True, output="[模拟输出] small_llm 推理结果",
        cost=0.10, quality=0.70, error_msg="",
    )

    # 案例1：执行失败 + 预算充足 -> RETRY（同质量重试）
    show_case(replanner, "案例1 [触发: 执行失败] 预算充足 $1.00",
              search, fail_result, [parse, llm, report], 1.00)

    # 案例2：质量不足 + 预算紧张 $0.10（不够 large_llm $0.50，够 small_llm $0.10）
    #        -> RETRY（降级低成本重试，质量要求放宽到 0.70）
    show_case(replanner, "案例2 [触发: 质量不足] 预算紧张 $0.10",
              llm, low_quality_result, [report], 0.10)

    # 案例3：执行失败 + 预算 $0.08（不够最便宜 LLM $0.10，但够后续报告 $0.05）
    #        -> SKIP（跳过并标记缺损）
    show_case(replanner, "案例3 [触发: 执行失败] 预算 $0.08，无法重试但可继续",
              llm, fail_result, [report], 0.08)

    # 案例4：执行失败 + 预算 $0.04（重试和继续后续都不够）-> ABORT
    show_case(replanner, "案例4 [触发: 执行失败] 预算 $0.04，无法补救",
              llm, fail_result, [report], 0.04)


if __name__ == "__main__":
    main()
