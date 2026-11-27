"""ExecutionAgent: perform the actual (simulated) service call.

执行员 Agent：负责真正调用服务 / 预订选项，并把结果回报给主 Agent。

职责边界（严格）：
- 执行前做「报价一致性校验」——黑板报价必须等于即将扣费的金额，
  不一致立即抛错（防止决策价与扣费价漂移）；
- 只执行、只上报；**不扣费**——扣费由主 Agent 独占，保证账本唯一写入。

包装对象可以是 SimulatedServiceRunner（企业任务）或
BookingRunner（行程预订），两者 run() 签名一致，故本 agent 无需区分。
"""

from __future__ import annotations

from ..simulated_services import ServiceResult
from ..task_planner import Subtask
from .base import Agent


class ExecutionAgent(Agent):
    """执行员：调用执行器，返回 ServiceResult。"""

    name = "ExecutionAgent"

    def __init__(self, runner) -> None:
        self._runner = runner

    def act(self, blackboard, service, subtask: Subtask) -> ServiceResult:
        """执行一次调用（签名与服务执行器一致，额外接收黑板）。"""
        quoted = blackboard.quoted_price(service.name)
        if quoted is not None and abs(quoted - service.cost_per_call) > 1e-9:
            raise ValueError(
                f"报价一致性校验失败：{service.name} 决策价 "
                f"${service.cost_per_call:.4f} ≠ 黑板报价 ${quoted:.4f}"
            )

        result = self._runner.run(service, subtask)
        if result.success:
            self._log(
                blackboard, "execute",
                f"{service.name} 成功，质量 {result.quality:.2f}，"
                f"开销 ${result.cost:.2f}",
            )
        else:
            self._log(
                blackboard, "execute",
                f"{service.name} 失败：{result.error_msg}（失败不扣费）",
            )
        return result
