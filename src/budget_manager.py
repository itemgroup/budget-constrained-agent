"""Budget tracking, spend deduction, logging.

预算账本：记录总预算、已消耗、剩余预算与每一步调用日志，
并提供开销预估与可执行性判断，供后续智能体执行循环调用。
"""

from __future__ import annotations

from dataclasses import dataclass

from .service_registry import Service


class InsufficientBudgetError(Exception):
    """剩余预算不足以支付本次服务调用时抛出。"""


@dataclass(frozen=True)
class CallRecord:
    """一次服务调用的记账记录。"""

    step: int             # 第几步（从 1 开始）
    service_name: str     # 调用的服务名
    service_type: str     # 服务类别
    cost: float           # 本次扣费（美元）
    remaining_after: float  # 扣费后的剩余预算


class BudgetLedger:
    """预算账本：跟踪预算消耗，判断下一步是否可执行。"""

    def __init__(self, total_budget: float) -> None:
        if total_budget < 0:
            raise ValueError("总预算不能为负数")
        self._total_budget: float = float(total_budget)
        self._spent: float = 0.0
        self._records: list[CallRecord] = []

    # ---------- 预算查询 ----------

    @property
    def total_budget(self) -> float:
        """总预算。"""
        return self._total_budget

    @property
    def spent(self) -> float:
        """已消耗金额。"""
        return self._spent

    @property
    def remaining(self) -> float:
        """剩余预算。"""
        return round(self._total_budget - self._spent, 10)

    @property
    def records(self) -> list[CallRecord]:
        """全部调用日志（返回副本，外部不可篡改内部状态）。"""
        return list(self._records)

    # ---------- 预估与可执行性判断 ----------

    def can_afford(self, cost: float) -> bool:
        """判断剩余预算是否足以支付一笔开销。"""
        return cost <= self.remaining

    def can_execute(self, service: Service) -> bool:
        """判断下一步调用该服务是否可执行（即预算足够支付单次调用）。"""
        return self.can_afford(service.cost_per_call)

    def estimate_cost(self, services: list[Service]) -> float:
        """预估一组后续调用的总开销。"""
        return round(sum(s.cost_per_call for s in services), 10)

    def estimate_plan(self, services: list[Service]) -> tuple[float, bool]:
        """预估后续计划的总开销，并判断当前剩余预算是否足以执行。

        返回 (预估总开销, 是否可执行)，供重规划/提前终止逻辑使用。
        """
        total = self.estimate_cost(services)
        return total, self.can_afford(total)

    # ---------- 记账 ----------

    def record_call(self, service: Service) -> CallRecord:
        """记录一次服务调用并扣费；预算不足时抛出 InsufficientBudgetError。"""
        if not self.can_execute(service):
            raise InsufficientBudgetError(
                f"预算不足：调用 {service.name} 需要 ${service.cost_per_call:.2f}，"
                f"剩余 ${self.remaining:.2f}"
            )
        self._spent = round(self._spent + service.cost_per_call, 10)
        record = CallRecord(
            step=len(self._records) + 1,
            service_name=service.name,
            service_type=service.service_type,
            cost=service.cost_per_call,
            remaining_after=self.remaining,
        )
        self._records.append(record)
        return record

    # ---------- 输出 ----------

    def summary(self) -> str:
        """生成账本消耗明细的文本摘要。"""
        lines = [
            f"总预算: ${self.total_budget:.2f} | "
            f"已消耗: ${self.spent:.2f} | "
            f"剩余: ${self.remaining:.2f}",
            "调用明细:",
        ]
        if not self._records:
            lines.append("  (暂无调用记录)")
        for r in self._records:
            lines.append(
                f"  步骤{r.step}: {r.service_name} [{r.service_type}] "
                f"-${r.cost:.2f} -> 剩余 ${r.remaining_after:.2f}"
            )
        return "\n".join(lines)
