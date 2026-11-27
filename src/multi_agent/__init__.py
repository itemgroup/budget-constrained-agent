"""Multi-agent layer (star topology + shared blackboard).

多智能体扩展包（proposal 范围之外的架构演进验证）：
不改动任何已有模块，把现有引擎能力包装成 5 个职责单一的 agent，
由主 Agent 以星型拓扑调度，全部通过共享黑板交换信息。

组成：
- Agent 基类 / Message          base.py
- Blackboard（唯一信息交换区）   blackboard.py
- ResearchAgent（查图记忆）      research.py
- PricingAgent（生成报价单）     pricing.py
- ExecutionAgent（执行/预订）    execution.py
- AuditorAgent（对账+校验+评分）  auditor.py
- MultiAgentOrchestrator（主）   orchestrator.py
"""

from .auditor import AuditReport, AuditorAgent
from .base import Agent, Message
from .blackboard import Blackboard
from .execution import ExecutionAgent
from .orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunReport,
    build_enterprise_multi_agent,
    build_travel_multi_agent,
)
from .pricing import PricingAgent
from .research import ResearchAgent

__all__ = [
    "Agent",
    "Message",
    "Blackboard",
    "ResearchAgent",
    "PricingAgent",
    "ExecutionAgent",
    "AuditorAgent",
    "AuditReport",
    "MultiAgentOrchestrator",
    "MultiAgentRunReport",
    "build_enterprise_multi_agent",
    "build_travel_multi_agent",
]
