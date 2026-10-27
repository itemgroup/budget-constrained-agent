"""Dynamic replan, failure handling, termination logic.

动态重规划模块：当子任务执行失败，或产出质量低于子任务最低要求时，决定补救策略。

策略优先级：
1. RETRY：预算允许则重试当前子任务——
   预算充足时按原质量要求重试；预算紧张时放宽质量要求、以更低成本服务重试；
2. SKIP：重试无望但剩余预算足以继续后续流程时，跳过当前子任务，标记任务存在缺损；
3. ABORT：剩余预算不足以补救（重试与继续后续都不可行）时，直接终止。

与 TaskPlanner 解耦：只调整后续子任务队列，不从原始大任务重新分解。
不接入真实 LLM / RAG。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .service_registry import ServiceRegistry
from .simulated_services import ServiceResult
from .task_planner import Subtask


class ReplanAction(Enum):
    """重规划动作。"""

    RETRY = "RETRY"  # 重试当前子任务（可降级到更低成本服务）
    SKIP = "SKIP"    # 跳过当前失败子任务，继续后续流程（任务存在缺损）
    ABORT = "ABORT"  # 剩余预算不足以补救，提前终止


@dataclass(frozen=True)
class ReplanResult:
    """结构化重规划结果。"""

    action: ReplanAction            # 重规划动作
    updated_subtasks: list[Subtask] # 调整后的子任务队列（ABORT 时为空）
    reason: str                     # 决策说明（用于透明日志）


class Replanner:
    """动态重规划器。服务注册表通过构造函数注入。"""

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    @staticmethod
    def should_trigger(subtask: Subtask, result: ServiceResult) -> bool:
        """判断是否触发重规划：执行失败，或产出质量低于子任务最低要求。"""
        return (not result.success) or (
            result.quality < subtask.required_min_quality
        )

    def _cheapest_cost(self, task_type: str) -> float | None:
        """某服务类别的最低单次价格；该类别无服务时返回 None。"""
        candidates = self._registry.list_by_type(task_type)
        return min((s.cost_per_call for s in candidates), default=None)

    def replan(
        self,
        current: Subtask,
        result: ServiceResult,
        remaining_subtasks: list[Subtask],
        remaining_budget: float,
    ) -> ReplanResult:
        """对当前子任务的失败/质量不足结果做重规划。

        参数:
            current: 当前子任务
            result: 本次执行结果（ServiceResult）
            remaining_subtasks: 尚未执行的后续子任务队列
            remaining_budget: 当前剩余预算
        返回:
            ReplanResult（action / updated_subtasks / reason）
        """
        cause = (
            "执行失败"
            if not result.success
            else f"质量不足（{result.quality:.2f} < 要求 {current.required_min_quality:.2f}）"
        )
        candidates = self._registry.list_by_type(current.task_type)

        # ---------- 1. 尝试 RETRY ----------
        if candidates:
            qualified_costs = [
                s.cost_per_call
                for s in candidates
                if s.quality_score >= current.required_min_quality
            ]
            min_qualified_cost = min(qualified_costs) if qualified_costs else None
            cheapest = min(candidates, key=lambda s: s.cost_per_call)

            # 预算充足：按原质量要求重试
            if min_qualified_cost is not None and remaining_budget >= min_qualified_cost:
                retry = replace(current, desc=current.desc + "（重试）")
                return ReplanResult(
                    ReplanAction.RETRY,
                    [retry, *remaining_subtasks],
                    f"{cause}；剩余预算 ${remaining_budget:.2f} 可支付同质量重试"
                    f"（最低 ${min_qualified_cost:.2f}），按原质量要求重试",
                )

            # 预算紧张：放宽质量要求，以更低成本服务重试
            if remaining_budget >= cheapest.cost_per_call:
                retry = replace(
                    current,
                    required_min_quality=cheapest.quality_score,
                    desc=current.desc + "（降级重试）",
                )
                return ReplanResult(
                    ReplanAction.RETRY,
                    [retry, *remaining_subtasks],
                    f"{cause}；预算不足以维持原质量要求，降级重试："
                    f"质量要求放宽至 {cheapest.quality_score:.2f}"
                    f"（可选 {cheapest.name}，${cheapest.cost_per_call:.2f}）",
                )

        # ---------- 2. RETRY 无望：SKIP 或 ABORT ----------
        # 计算继续后续流程的最低开销（任一后续类别无服务则视为不可继续）
        continuation_cost = 0.0
        for st in remaining_subtasks:
            c = self._cheapest_cost(st.task_type)
            if c is None:
                continuation_cost = float("inf")
                break
            continuation_cost += c

        if remaining_subtasks and remaining_budget >= continuation_cost:
            return ReplanResult(
                ReplanAction.SKIP,
                list(remaining_subtasks),
                f"{cause}；剩余预算 ${remaining_budget:.2f} 不足以重试当前子任务，"
                f"跳过并继续后续流程（任务存在缺损：缺少 [{current.task_type}] 结果）",
            )

        return ReplanResult(
            ReplanAction.ABORT,
            [],
            f"{cause}；剩余预算 ${remaining_budget:.2f} 不足以补救"
            f"（继续后续最低需 ${continuation_cost:.2f}），提前终止",
        )
