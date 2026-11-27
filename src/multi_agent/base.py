"""Agent base class and shared message model for the multi-agent layer.

多智能体基类：每个 worker 只做一件事，通过读写黑板（Blackboard）协作，
彼此之间不直接通信。

设计约束（详见架构图）：
- worker 只上报结果，不直接扣费——扣费由主 Agent 独占（账本不变量）；
- 真实 LLM 调用可折算成「服务计费」走同一个 BudgetLedger，
  让多 agent 的协作开销被预算体系捕获（默认关闭，避免影响 A/B 对照）。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..service_registry import Service

#: 真实 LLM 单次调用折算的服务计费项（仅当 bill_llm_calls=True 时扣费）
LLM_CALL_SERVICE = Service(
    name="deepseek_chat",
    service_type="LLM Reasoning",
    cost_per_call=0.001,
    quality_score=0.90,
    description="DeepSeek chat call (billed per invocation)",
)


@dataclass(frozen=True)
class Message:
    """黑板上的一条 agent 活动日志。"""

    agent: str      # 哪个 agent
    action: str     # 动作类型：lookup / quote / execute / audit / decompose
    detail: str     # 人类可读的说明
    cost: float = 0.0   # 本次动作产生的计费（LLM 折算）
    ts: float = 0.0     # 时间戳（仅用于排序展示）


class Agent(ABC):
    """所有 worker 的抽象基类。"""

    name: str = "Agent"

    @abstractmethod
    def act(self, blackboard) -> None:
        """执行本 agent 的职责，结果写入黑板。"""

    # ---------- 通用助手 ----------

    def _log(
        self, blackboard, action: str, detail: str, cost: float = 0.0
    ) -> None:
        """向黑板追加一条活动日志。"""
        blackboard.messages.append(
            Message(self.name, action, detail, round(cost, 6), time.time())
        )

    def _bill_llm(self, blackboard, calls: int = 1) -> float:
        """把真实 LLM 调用折算成服务计费，走同一个账本。

        默认关闭（blackboard.bill_llm_calls=False），保证模拟环境下的
        多 agent 与单 agent 结果完全可比；开启后协作开销进入账本。
        预算不足时静默跳过，不抛异常。
        """
        if not getattr(blackboard, "bill_llm_calls", False):
            return 0.0
        billed = 0.0
        for _ in range(calls):
            if blackboard.ledger.can_execute(LLM_CALL_SERVICE):
                blackboard.ledger.record_call(LLM_CALL_SERVICE)
                billed += LLM_CALL_SERVICE.cost_per_call
        return round(billed, 6)
