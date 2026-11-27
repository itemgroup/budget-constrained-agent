"""Shared blackboard: the single information exchange area (star topology hub).

黑板是星型拓扑的物理中心：所有 agent 只与黑板交互，不互相私聊。
这保证了三件事：
- 单一数据源：选择器 / 执行器 / 账本看到的是同一份报价单，决策价 == 扣费价；
- 单一写入权：price_sheet 由 PricingAgent 独占写，research_notes 由
  ResearchAgent 独占写，避免多份拷贝漂移；
- 可观测：messages 记录每一次 agent 动作，运行报告可完整回放协作过程。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..agent_core import StepLog, SubtaskOutput
from ..budget_manager import BudgetLedger
from ..memory import ExperienceGraph
from ..service_registry import ServiceRegistry
from ..task_planner import Subtask
from .base import Message

if TYPE_CHECKING:  # 仅类型检查时需要，避免与 auditor 形成循环导入
    from .auditor import AuditReport


@dataclass
class Blackboard:
    """一次运行内所有 agent 共享的中央数据区。"""

    task: str                                   # 原始任务
    ledger: BudgetLedger                        # 全局唯一账本（主 Agent 独占写入）

    # ---- 由主 Agent 装配 ----
    registry: ServiceRegistry | None = None      # 本次生效的价目表（企业=8 服务 / 行程=报价单）
    memory: ExperienceGraph | None = None        # 图记忆（Research 读，主 Agent 回填）

    # ---- 由各 worker 独占写入 ----
    plan: list[Subtask] = field(default_factory=list)          # 主 Agent：分解链
    price_sheet: dict[str, float] = field(default_factory=dict)  # Pricing：本次报价
    research_notes: dict[str, Any] = field(default_factory=dict)  # Research：调研结论
    audit_report: "AuditReport | None" = None                   # Auditor：审计结论

    # ---- 由主 Agent 追加 ----
    outputs: list[SubtaskOutput] = field(default_factory=list)
    steps: list[StepLog] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    # ---- 全队可追加 ----
    messages: list[Message] = field(default_factory=list)

    # ---- 配置 ----
    bill_llm_calls: bool = False                 # 是否把 LLM 调用计入账本

    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    # ---------- 便捷方法 ----------

    def log(self, agent: str, action: str, detail: str, cost: float = 0.0) -> None:
        """追加一条活动日志（供不在 Agent 基类里的调用方使用）。"""
        self.messages.append(Message(agent, action, detail, round(cost, 6), 0.0))

    def quoted_price(self, service_name: str) -> float | None:
        """查询某个服务的本次报价；未报价返回 None。"""
        return self.price_sheet.get(service_name)

    def messages_of(self, agent: str) -> list[Message]:
        """按 agent 过滤活动日志。"""
        return [m for m in self.messages if m.agent == agent]
