"""Score & rank candidate services.

预算感知服务选择器：在剩余预算约束下为子任务挑选最合适的服务。

选择策略：
1. 在【满足子任务最低质量要求】且【剩余预算可支付】的服务中，选质量分最高者；
2. 若满足质量要求的服务都支付不起 -> 预算紧张，自动降级：
   放宽最低质量要求，在可支付的服务中选质量最高者；
3. 若该类别下所有服务都支付不起 -> 返回预算不足标记。

与 BudgetLedger 解耦：本模块只接收"剩余预算"数值参数，
不直接读取账本对象；ServiceRegistry 通过构造函数注入。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .service_registry import Service, ServiceRegistry
from .task_planner import Subtask


class SelectionStatus(Enum):
    """一次服务选择的状态。"""

    OK = "ok"                                  # 正常选中（满足最低质量要求）
    DOWNGRADED = "downgraded"                  # 预算紧张，已降级（质量低于要求）
    INSUFFICIENT_BUDGET = "insufficient_budget"  # 预算不足，无可用服务


@dataclass(frozen=True)
class SelectionResult:
    """一次服务选择的结果。"""

    status: SelectionStatus  # 选择状态
    service: Service | None  # 选中的服务；预算不足时为 None
    reason: str              # 决策说明（用于透明日志）


class BudgetAwareSelector:
    """预算感知服务选择器。"""

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    def select(self, subtask: Subtask, remaining_budget: float) -> SelectionResult:
        """为子任务选择服务。

        参数:
            subtask: 当前子任务（提供 task_type 与 required_min_quality）
            remaining_budget: 当前剩余预算（美元）
        返回:
            SelectionResult（含状态、选中服务、决策说明）
        """
        candidates = self._registry.list_by_type(subtask.task_type)
        if not candidates:
            return SelectionResult(
                SelectionStatus.INSUFFICIENT_BUDGET,
                None,
                f"服务注册表中没有 {subtask.task_type} 类别的服务",
            )

        # 第一优先级：满足最低质量要求 + 预算可支付 -> 最大化质量
        qualified = [
            s for s in candidates
            if s.quality_score >= subtask.required_min_quality
        ]
        affordable_qualified = [
            s for s in qualified if s.cost_per_call <= remaining_budget
        ]
        if affordable_qualified:
            # 质量分优先，同分时取价格更低者
            best = max(
                affordable_qualified,
                key=lambda s: (s.quality_score, -s.cost_per_call),
            )
            return SelectionResult(
                SelectionStatus.OK,
                best,
                f"满足最低质量 {subtask.required_min_quality:.2f}，"
                f"在可支付服务中选质量最高者 {best.name}",
            )

        # 预算紧张：放宽最低质量要求，降级到可支付服务中质量最高者
        affordable = [s for s in candidates if s.cost_per_call <= remaining_budget]
        if affordable:
            best = max(
                affordable,
                key=lambda s: (s.quality_score, -s.cost_per_call),
            )
            return SelectionResult(
                SelectionStatus.DOWNGRADED,
                best,
                f"预算紧张：满足质量 {subtask.required_min_quality:.2f} 的服务均不可支付，"
                f"降级为 {best.name}（质量 {best.quality_score:.2f}）",
            )

        # 该类别下所有服务都支付不起
        cheapest_cost = min(s.cost_per_call for s in candidates)
        return SelectionResult(
            SelectionStatus.INSUFFICIENT_BUDGET,
            None,
            f"预算不足：{subtask.task_type} 类别最低价格为 ${cheapest_cost:.2f}，"
            f"剩余预算 ${remaining_budget:.2f}",
        )
