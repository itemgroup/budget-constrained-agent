"""MultiAgentOrchestrator: the main agent (hub) that dispatches to workers.

主 Agent（总指挥）：维护唯一决策权、唯一账本写入权、唯一循环控制权。

与单 agent 版（src/agent_core.py）的关系：
- 状态机（AgentState 五态）、StepLog / SubtaskOutput / RunReport 语义完全复用；
- 唯一的区别是「谁干活」：选服务与记账仍由本类拍板，
  执行派给 ExecutionAgent，自动化流程由 Research / Pricing / Auditor 分担。

一次运行的协作顺序：
    ResearchAgent.lookup  ->  分解（命中记忆则零 LLM）  ->  PricingAgent.quote
      ->  主循环（选择 -> ExecutionAgent.execute -> 记账 -> 重规划）
      ->  AuditorAgent.audit（对账 + 格式 + 评分，产出合规交付物）
      ->  回填图记忆

所有 agent 只读写黑板；扣费只发生在主循环这一处。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent_core import AgentState, StepLog, SubtaskOutput
from ..budget_manager import BudgetLedger
from ..replanner import ReplanAction, Replanner
from ..scenarios.scenario import (
    BookingRunner,
    ScenarioPlanner,
    build_price_registry,
    load_catalog,
)
from ..service_registry import ServiceRegistry
from ..service_selector import BudgetAwareSelector, SelectionStatus
from ..simulated_services import SimulatedServiceRunner
from ..task_planner import Subtask, TaskPlanner
from .auditor import AuditReport, AuditorAgent
from .base import Message
from .blackboard import Blackboard
from .execution import ExecutionAgent
from .pricing import PricingAgent
from .research import ResearchAgent


@dataclass(frozen=True)
class MultiAgentRunReport:
    """多 agent 运行报告（字段与 agent_core.RunReport 兼容，额外附带协作信息）。"""

    task: str
    state: AgentState
    termination_reason: str
    total_budget: float
    spent: float
    remaining: float
    steps: list[StepLog]
    outputs: list[SubtaskOutput]
    skipped_subtasks: list[str]
    ledger_summary: str
    audit_report: AuditReport | None        # 审计结论
    messages: list[Message]                 # 全队活动日志
    research_notes: dict                    # 调研结论（记忆是否命中）
    plan_size: int                          # 计划子任务数


class MultiAgentOrchestrator:
    """多 agent 主调度器：派活、记账、控制循环、汇总审计。"""

    def __init__(
        self,
        planner,
        runner,
        pricing: PricingAgent | None = None,
        registry: ServiceRegistry | None = None,
        memory=None,
        judge=None,
        max_steps: int = 20,
        bill_llm_calls: bool = False,
        selector=None,
        replanner=None,
    ) -> None:
        """装配主 Agent。

        参数:
            planner: 分解器（TaskPlanner / LLMTaskPlanner / ScenarioPlanner），
                     需实现 decompose(task) -> list[Subtask]
            runner: 执行器（SimulatedServiceRunner / BookingRunner），
                    需实现 run(service, subtask) -> ServiceResult
            pricing: 询价员；缺省用给定 registry 的基准价
            registry: 价目表（pricing 若带 catalog 则以其生成的报价单为准）
            memory: 图记忆 ExperienceGraph（可选）
            judge: 评分器（LLMJudge / RealLLMJudge），供 AuditorAgent 使用
            max_steps: 主循环最大轮次（防死循环）
            bill_llm_calls: 是否把 LLM 调用折算成服务计费（默认关闭）
            selector: 服务选择策略（缺省 BudgetAwareSelector）；实验对比可注入基线策略
            replanner: 重规划策略（缺省 Replanner）；实验对比可注入空重规划
        """
        self._planner = planner
        self._runner = runner
        self._pricing = (
            pricing if pricing is not None else PricingAgent(registry=registry)
        )
        self._registry = registry
        self._memory = memory
        self._judge = judge
        self._max_steps = max_steps
        self._bill = bill_llm_calls
        self._selector = selector
        self._replanner = replanner

        # 全局执行上下文（run 后填充，便于实验脚本读取）
        self.state: AgentState = AgentState.IDLE
        self.blackboard: Blackboard | None = None

    # ---------- 兼容实验脚本的只读属性 ----------

    @property
    def steps(self) -> list[StepLog]:
        """最近一次运行的执行日志（与 agent_core 同名属性兼容）。"""
        return list(self.blackboard.steps) if self.blackboard else []

    @property
    def outputs(self) -> list[SubtaskOutput]:
        return list(self.blackboard.outputs) if self.blackboard else []

    # ---------- 主入口 ----------

    def run(self, task: str, total_budget: float) -> MultiAgentRunReport:
        """执行完整任务链路：调研 -> 分解 -> 询价 -> 主循环 -> 审计。"""
        self.state = AgentState.RUNNING
        bb = Blackboard(
            task=task,
            ledger=BudgetLedger(total_budget),
            registry=self._registry,
            memory=self._memory,
            bill_llm_calls=self._bill,
        )
        self.blackboard = bb

        research = ResearchAgent(self._memory)
        execution = ExecutionAgent(self._runner)
        auditor = AuditorAgent(self._judge)

        # ---- ① 调研：查经验记忆 ----
        research.act(bb)

        # ---- ② 询价：生成本次报价单（先于分解，便于校验记忆方案是否可用） ----
        self._pricing.act(bb)
        registry = bb.registry
        if registry is None:  # 理论上 pricing.act 已填充；兜底
            registry = ServiceRegistry()
            bb.registry = registry

        # ---- ③ 分解：记忆命中且任务类型在当前价目表可用 -> 复用，零 LLM ----
        notes = bb.research_notes
        memory_plan = notes.get("plan") if notes.get("hit") else None
        if memory_plan and all(
            registry.list_by_type(st.task_type) for st in memory_plan
        ):
            plan = list(memory_plan)
            bb.log(
                self.__class__.__name__, "decompose",
                f"复用经验记忆的 {len(plan)} 步分解链（分解阶段零 LLM 调用）",
            )
        else:
            plan = self._planner.decompose(task)
            if memory_plan:
                bb.log(
                    self.__class__.__name__, "decompose",
                    "记忆方案中的类别在当前价目表不可用，改用现场分解",
                )
            else:
                bb.log(
                    self.__class__.__name__, "decompose",
                    f"现场分解得到 {len(plan)} 个子任务",
                )
            if self._memory is not None:
                self._memory.add_experience(task, plan)
        bb.plan = list(plan)

        # ---- ④ 主循环：选择（主 Agent）-> 执行（Execution）-> 记账（主 Agent） ----
        selector = self._selector or BudgetAwareSelector(registry)
        replanner = self._replanner or Replanner(registry)
        queue: list[Subtask] = list(plan)
        state = AgentState.RUNNING
        termination_reason = "所有子任务执行完毕"

        while queue:
            if len(bb.steps) >= self._max_steps:
                state = AgentState.FAILED
                termination_reason = (
                    f"达到最大轮次上限 {self._max_steps}，强制终止（防死循环）"
                )
                break
            if bb.ledger.remaining <= 0:
                state = AgentState.BUDGET_EXHAUSTED
                termination_reason = "预算耗尽，仍有子任务未执行"
                break

            current = queue[0]
            step_no = len(bb.steps) + 1

            # 选择服务（唯一决策点）
            sel = selector.select(current, bb.ledger.remaining)
            if sel.status is SelectionStatus.INSUFFICIENT_BUDGET:
                bb.steps.append(StepLog(
                    step_no, current.desc, current.task_type, "",
                    sel.status.value, False, None, 0.0, 0.0, sel.reason,
                ))
                state = AgentState.BUDGET_EXHAUSTED
                termination_reason = sel.reason
                break

            # 派给执行员
            result = execution.act(bb, sel.service, current)

            # 记账（唯一扣费点）
            if result.success:
                bb.ledger.record_call(sel.service)

            # 验收（降级方案执行成功即接受，避免无效重复扣费）
            downgraded_accept = (
                sel.status is SelectionStatus.DOWNGRADED and result.success
            )
            if downgraded_accept or not Replanner.should_trigger(current, result):
                note = (
                    f"{sel.reason}；降级方案执行成功，接受结果"
                    if downgraded_accept
                    else f"{sel.reason}；质量达标，子任务完成"
                )
                bb.outputs.append(SubtaskOutput(
                    current.desc, current.task_type, sel.service.name,
                    result.output, result.quality,
                ))
                bb.steps.append(StepLog(
                    step_no, current.desc, current.task_type, sel.service.name,
                    sel.status.value, True, True, result.cost, result.quality, note,
                ))
                queue.pop(0)
                continue

            # 失败 / 质量不足 -> 规则版重规划（确定性，不吃 token）
            cause = (
                result.error_msg
                if not result.success
                else f"质量不足（{result.quality:.2f} < 要求 "
                     f"{current.required_min_quality:.2f}）"
            )
            replan = replanner.replan(
                current, result, queue[1:], bb.ledger.remaining
            )
            bb.steps.append(StepLog(
                step_no, current.desc, current.task_type, sel.service.name,
                sel.status.value, True, result.success, result.cost,
                result.quality,
                f"{cause}；重规划 {replan.action.value}: {replan.reason}",
            ))
            bb.log(
                "Replanner", replan.action.value,
                f"{current.desc}：{replan.reason}",
            )
            if replan.action is ReplanAction.ABORT:
                state = AgentState.BUDGET_EXHAUSTED
                termination_reason = replan.reason
                queue = replan.updated_subtasks
                break
            if replan.action is ReplanAction.SKIP:
                bb.skipped.append(current.desc)
            queue = replan.updated_subtasks
        else:
            state = AgentState.COMPLETED

        # ---- ⑤ 审计：对账 + 格式 + 评分，产出合规交付物 ----
        auditor.act(bb)

        # ---- ⑥ 回填图记忆（图越用越准） ----
        if self._memory is not None:
            self._memory.record_run_outputs(task, bb.steps)

        self.state = state
        return MultiAgentRunReport(
            task=task,
            state=state,
            termination_reason=termination_reason,
            total_budget=bb.ledger.total_budget,
            spent=bb.ledger.spent,
            remaining=bb.ledger.remaining,
            steps=list(bb.steps),
            outputs=list(bb.outputs),
            skipped_subtasks=list(bb.skipped),
            ledger_summary=bb.ledger.summary(),
            audit_report=bb.audit_report,
            messages=list(bb.messages),
            research_notes=dict(bb.research_notes),
            plan_size=len(bb.plan),
        )


# ---------- 一键装配（企业任务 / 行程规划） ----------

def build_enterprise_multi_agent(
    seed: int = 0,
    planner=None,
    memory=None,
    judge=None,
    max_steps: int = 20,
    bill_llm_calls: bool = False,
    selector=None,
    replanner=None,
    registry: ServiceRegistry | None = None,
    runner=None,
) -> MultiAgentOrchestrator:
    """装配企业任务多 agent 系统（8 个模拟服务，proposal 主场景）。"""
    registry = registry if registry is not None else ServiceRegistry()
    return MultiAgentOrchestrator(
        planner=planner if planner is not None else TaskPlanner(),
        runner=runner if runner is not None else SimulatedServiceRunner(seed=seed),
        pricing=PricingAgent(registry=registry),
        registry=registry,
        memory=memory,
        judge=judge,
        max_steps=max_steps,
        bill_llm_calls=bill_llm_calls,
        selector=selector,
        replanner=replanner,
    )


def build_travel_multi_agent(
    seed: int = 0,
    catalog_path=None,
    memory=None,
    judge=None,
    max_steps: int = 20,
) -> MultiAgentOrchestrator:
    """装配行程规划多 agent 系统（实时报价 + 预订语义，场景扩展包）。"""
    catalog = load_catalog() if catalog_path is None else load_catalog(catalog_path)
    registry, rates = build_price_registry(catalog, seed=seed)
    return MultiAgentOrchestrator(
        planner=ScenarioPlanner(catalog, registry),
        runner=BookingRunner(rates, seed=seed),
        pricing=PricingAgent(
            registry=registry,
            label=f"行程场景报价单（基准价 ±15% 波动，seed={seed}）",
        ),
        registry=registry,
        memory=memory,
        judge=judge,
        max_steps=max_steps,
    )
