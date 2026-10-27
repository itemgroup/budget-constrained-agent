"""集成 Demo：企业任务 -> 拆分子任务 -> 预算感知选服务 -> 写入账本。

演示 TaskPlanner + BudgetAwareSelector + BudgetLedger + ServiceRegistry
四个模块的对接。以下为固定顺序的演示流程，不是完整 Agent 循环。

运行方式（项目根目录下）：
    python demo_plan_select.py
"""

from src.budget_manager import BudgetLedger
from src.service_registry import ServiceRegistry
from src.service_selector import BudgetAwareSelector, SelectionStatus
from src.task_planner import TaskPlanner


def run_scenario(title: str, task: str, total_budget: float) -> None:
    """跑一个完整场景：分解 -> 逐步选择并记账 -> 输出账本明细。"""
    print("=" * 64)
    print(f"{title}（总预算 ${total_budget:.2f}）")
    print(f"企业任务: {task}")
    print("=" * 64)

    planner = TaskPlanner()
    registry = ServiceRegistry()
    selector = BudgetAwareSelector(registry)
    ledger = BudgetLedger(total_budget)

    # 1. 任务分解
    subtasks = planner.decompose(task)
    print("\n[1] 任务分解结果:")
    for st in subtasks:
        print(
            f"  子任务{st.order}: {st.desc} [{st.task_type}] "
            f"最低质量 {st.required_min_quality:.2f}，预估 ${st.estimated_cost:.2f}"
        )
    plan_est = round(sum(st.estimated_cost for st in subtasks), 10)
    print(f"  计划总预估开销: ${plan_est:.2f}")

    # 2. 逐步：选择服务 -> 写入账本
    print("\n[2] 逐步选择服务并记账:")
    for st in subtasks:
        result = selector.select(st, ledger.remaining)
        if result.status is SelectionStatus.INSUFFICIENT_BUDGET:
            print(f"  子任务{st.order} [{st.task_type}] 终止: {result.reason}")
            break
        record = ledger.record_call(result.service)
        tag = "正常" if result.status is SelectionStatus.OK else "降级"
        print(
            f"  子任务{st.order} [{st.task_type}] {tag}选中 {record.service_name} "
            f"(质量 {result.service.quality_score:.2f}) "
            f"-${record.cost:.2f} -> 剩余 ${record.remaining_after:.2f}"
        )
        print(f"      决策说明: {result.reason}")

    # 3. 账本消耗明细
    print("\n[3] 账本明细:")
    print(ledger.summary())
    print()


def main() -> None:
    task = "调研文档，分析要点并生成摘要报告"
    # 场景一：预算充足，每一步都应选最高质量服务
    run_scenario("场景一：预算充足", task, 1.00)
    # 场景二：预算紧张，演示自动降级与预算不足终止
    run_scenario("场景二：预算紧张", task, 0.25)


if __name__ == "__main__":
    main()
