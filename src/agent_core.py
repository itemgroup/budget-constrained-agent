"""Main agent loop: decompose task -> select service -> execute -> feedback.

智能体核心调度器：串联任务分解、服务选择、模拟执行、账本记账与动态重规划，
维护执行状态机，驱动完整任务链路直到终止。

工作流：接收原始任务 + 总预算 -> 任务分解 -> 循环处理子任务：
  选择服务 -> 校验选择结果 -> 模拟执行 -> 成功记账并保存结果 ->
  质量达标则继续；失败/质量不足则调用重规划（RETRY / SKIP / ABORT）。

终止条件：子任务队列清空（COMPLETED）、预算耗尽或 ABORT（BUDGET_EXHAUSTED）、
达到最大轮次上限（FAILED，防死循环）。

仅做编排与状态维护；下层模块全部通过构造函数注入，不修改其代码。
不接入真实 LLM / RAG。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .budget_manager import BudgetLedger
from .replanner import Replanner, ReplanAction
from .service_registry import ServiceRegistry
from .service_selector import BudgetAwareSelector, SelectionStatus
from .simulated_services import SimulatedServiceRunner
from .task_planner import Subtask, TaskPlanner


class AgentState(Enum):
    """智能体执行状态机。"""

    IDLE = "idle"                            # 未启动
    RUNNING = "running"                      # 执行中
    COMPLETED = "completed"                  # 全部子任务完成
    BUDGET_EXHAUSTED = "budget_exhausted"    # 预算耗尽 / ABORT 终止
    FAILED = "failed"                        # 异常终止（如达到最大轮次）


@dataclass(frozen=True)
class StepLog:
    """一步循环的执行日志。"""

    step: int               # 第几轮（从 1 开始）
    subtask_desc: str       # 当前子任务描述
    task_type: str          # 服务类别
    service_name: str       # 选中服务名；未选出时为空字符串
    selection_status: str   # ok / downgraded / insufficient_budget
    executed: bool          # 是否真正调用了服务
    success: bool | None    # 执行是否成功；未执行时为 None
    cost: float             # 本次实际开销
    quality: float          # 本次产出质量分
    note: str               # 选择原因 / 重规划决策 / 终止原因


@dataclass(frozen=True)
class SubtaskOutput:
    """一个成功完成的子任务的输出结果。"""

    subtask_desc: str
    task_type: str
    service_name: str
    output: str
    quality: float


@dataclass(frozen=True)
class RunReport:
    """完整运行报告。"""

    task: str                   # 原始任务
    state: AgentState           # 终止时状态
    termination_reason: str     # 终止原因
    total_budget: float
    spent: float
    remaining: float
    steps: list[StepLog]            # 每一步执行日志
    outputs: list[SubtaskOutput]    # 所有成功子任务的结果
    skipped_subtasks: list[str]     # 被 SKIP 的子任务（任务缺损标记）
    ledger_summary: str             # 账本消耗明细


class AgentOrchestrator:
    """智能体核心调度器：维护全局执行上下文并驱动主循环。"""

    def __init__(
        self,
        planner: TaskPlanner | None = None,
        selector: BudgetAwareSelector | None = None,
        runner: SimulatedServiceRunner | None = None,
        replanner: Replanner | None = None,
        registry: ServiceRegistry | None = None,
        max_steps: int = 20,
    ) -> None:
        # 下层模块全部依赖注入，缺省时自动装配默认实现
        self._registry = registry if registry is not None else ServiceRegistry()
        self._planner = planner if planner is not None else TaskPlanner()
        self._selector = (
            selector if selector is not None else BudgetAwareSelector(self._registry)
        )
        self._runner = runner if runner is not None else SimulatedServiceRunner()
        self._replanner = (
            replanner if replanner is not None else Replanner(self._registry)
        )
        self._max_steps = max_steps

        # 全局执行上下文（run 时填充）
        self.state: AgentState = AgentState.IDLE
        self.task: str = ""
        self.queue: list[Subtask] = []
        self.ledger: BudgetLedger | None = None
        self.steps: list[StepLog] = []
        self.outputs: list[SubtaskOutput] = []
        self.skipped: list[str] = []

    def run(self, task: str, total_budget: float) -> RunReport:
        """主入口：执行完整任务链路，返回运行报告。"""
        self.state = AgentState.RUNNING
        self.task = task
        self.ledger = BudgetLedger(total_budget)
        self.queue = self._planner.decompose(task)
        self.steps, self.outputs, self.skipped = [], [], []
        termination_reason = "所有子任务执行完毕"

        while self.queue:
            # ---- 终止条件：最大轮次 / 预算耗尽 ----
            if len(self.steps) >= self._max_steps:
                self.state = AgentState.FAILED
                termination_reason = (
                    f"达到最大轮次上限 {self._max_steps}，强制终止（防死循环）"
                )
                break
            if self.ledger.remaining <= 0:
                self.state = AgentState.BUDGET_EXHAUSTED
                termination_reason = "预算耗尽，仍有子任务未执行"
                break

            current = self.queue[0]
            step_no = len(self.steps) + 1

            # ---- 1-2. 选择服务并校验选择结果 ----
            sel = self._selector.select(current, self.ledger.remaining)
            if sel.status is SelectionStatus.INSUFFICIENT_BUDGET:
                self.steps.append(StepLog(
                    step_no, current.desc, current.task_type, "",
                    sel.status.value, False, None, 0.0, 0.0, sel.reason,
                ))
                self.state = AgentState.BUDGET_EXHAUSTED
                termination_reason = sel.reason
                break

            # ---- 3. 调用模拟服务执行 ----
            result = self._runner.run(sel.service, current)

            # ---- 4. 成功则记账 ----
            if result.success:
                self.ledger.record_call(sel.service)

            # ---- 5. 质量验收：达标则保存结果，继续下一个子任务 ----
            # DOWNGRADED 是预算约束下的主动降级决策：执行成功即接受结果，
            # 不再因"质量低于原要求"触发重规划
            # （重试同类低成本服务不会得到更高质量，只会重复扣费）；
            # 执行失败（success=False）无论是否降级都触发重规划。
            downgraded_accept = (
                sel.status is SelectionStatus.DOWNGRADED and result.success
            )
            if downgraded_accept or not Replanner.should_trigger(current, result):
                note = (
                    f"{sel.reason}；降级方案执行成功，接受结果"
                    if downgraded_accept
                    else f"{sel.reason}；质量达标，子任务完成"
                )
                self.outputs.append(SubtaskOutput(
                    current.desc, current.task_type, sel.service.name,
                    result.output, result.quality,
                ))
                self.steps.append(StepLog(
                    step_no, current.desc, current.task_type, sel.service.name,
                    sel.status.value, True, True, result.cost, result.quality,
                    note,
                ))
                self.queue.pop(0)
                continue

            # ---- 6. 失败/质量不足：调用重规划，按动作更新队列 ----
            cause = (
                result.error_msg
                if not result.success
                else f"质量不足（{result.quality:.2f} < 要求 "
                     f"{current.required_min_quality:.2f}）"
            )
            plan = self._replanner.replan(
                current, result, self.queue[1:], self.ledger.remaining
            )
            self.steps.append(StepLog(
                step_no, current.desc, current.task_type, sel.service.name,
                sel.status.value, True, result.success, result.cost,
                result.quality,
                f"{cause}；重规划 {plan.action.value}: {plan.reason}",
            ))
            if plan.action is ReplanAction.ABORT:
                self.state = AgentState.BUDGET_EXHAUSTED
                termination_reason = plan.reason
                self.queue = plan.updated_subtasks
                break
            if plan.action is ReplanAction.SKIP:
                self.skipped.append(current.desc)
            self.queue = plan.updated_subtasks  # RETRY: 重试项已置于队首
        else:
            # while 正常结束（队列清空）
            self.state = AgentState.COMPLETED

        return RunReport(
            task=self.task,
            state=self.state,
            termination_reason=termination_reason,
            total_budget=self.ledger.total_budget,
            spent=self.ledger.spent,
            remaining=self.ledger.remaining,
            steps=list(self.steps),
            outputs=list(self.outputs),
            skipped_subtasks=list(self.skipped),
            ledger_summary=self.ledger.summary(),
        )
