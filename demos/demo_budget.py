"""Demo：手动模拟 3 次服务调用，打印预算账本消耗明细。

运行方式（在项目根目录下）：
    python demos/demo_budget.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.budget_manager import BudgetLedger
from src.service_registry import ServiceRegistry


def main() -> None:
    # 1. 初始化：服务注册表（内置 proposal 3.2 的 8 个服务）+ 总预算 $1.00 的账本
    registry = ServiceRegistry()
    ledger = BudgetLedger(total_budget=1.00)

    # 2. 手动指定 3 次调用：检索 -> LLM 推理 -> 报告生成
    planned = ["premium_search", "large_llm", "custom_report"]

    print("=" * 60)
    print("预算账本 Demo：3 次模拟服务调用（总预算 $1.00）")
    print("=" * 60)

    for name in planned:
        service = registry.get(name)
        # 调用前先判断预算是否足够支付本次调用
        if not ledger.can_execute(service):
            print(f"预算不足，跳过: {service.name}")
            continue
        record = ledger.record_call(service)
        print(
            f"步骤{record.step} 调用 {record.service_name:<15} "
            f"[{record.service_type}] 扣费 ${record.cost:.2f}，"
            f"剩余 ${record.remaining_after:.2f}"
        )

    # 3. 预估一个后续计划的总开销，并判断当前剩余预算是否可执行
    follow_up = [registry.get(n) for n in ("basic_search", "small_llm", "template_report")]
    est, affordable = ledger.estimate_plan(follow_up)
    verdict = "可执行" if affordable else "不可执行（预算不足）"
    print(f"\n后续计划预估开销: ${est:.2f} -> 当前剩余预算{verdict}")

    # 4. 打印账本消耗明细
    print("\n" + ledger.summary())


if __name__ == "__main__":
    main()
