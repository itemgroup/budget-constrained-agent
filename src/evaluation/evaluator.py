"""Metrics, baseline runner, ablation runner.

评估模块（proposal 3.4-3.6）：

- 指标（3.5）：预算约束下任务成功率、平均模拟开销、超预算比例、
  工具调用次数（含冗余重复调用）、LLM-Judge 评分；
- 三个基线（3.4）：Standard / Minimum-Cost / Budget Hard-Stop，
  以及 Proposed Agent（完整版）；
- 消融变体（3.6）：分别禁用预算感知选择模块、动态重规划模块。

基线与消融通过【注入不同的选择器/重规划器】实现，
不修改任何下层模块代码。所有运行均为模拟，不接入真实 LLM / RAG。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..agent_core import AgentState, RunReport
from ..multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    build_enterprise_multi_agent,
)
from ..replanner import ReplanAction, ReplanResult
from ..service_registry import ServiceRegistry
from ..service_selector import (
    BudgetAwareSelector,
    SelectionResult,
    SelectionStatus,
)
from ..simulated_services import ServiceResult
from ..task_planner import Subtask, TaskPlanner
from .llm_judge import LLMJudge

# Standard Agent 无预算约束（proposal 3.4）：内部给一个极大预算，
# 超预算比例仍以该任务的预算档位为参照计算。
STANDARD_INTERNAL_BUDGET = 999.0

AGENT_NAMES = ("standard", "min_cost", "hard_stop", "proposed")
ABLATION_NAMES = ("full", "no_budget_aware_selection", "no_dynamic_replanning")

_DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "test_tasks.json"


# ---------- 基线用选择器变体（仅实现接口，不改动原选择器） ----------

class _QualityFirstSelector:
    """Standard Agent 用：不看预算，永远选质量最高者。"""

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    def select(self, subtask: Subtask, remaining_budget: float) -> SelectionResult:
        candidates = self._registry.list_by_type(subtask.task_type)
        if not candidates:
            return SelectionResult(
                SelectionStatus.INSUFFICIENT_BUDGET, None,
                f"服务注册表中没有 {subtask.task_type} 类别的服务",
            )
        best = max(candidates, key=lambda s: s.quality_score)
        return SelectionResult(
            SelectionStatus.OK, best,
            f"无预算约束，直接选质量最高者 {best.name}",
        )


class _MinCostSelector:
    """Minimum-Cost Agent 用：永远选可支付的最便宜服务，不看质量。"""

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    def select(self, subtask: Subtask, remaining_budget: float) -> SelectionResult:
        candidates = self._registry.list_by_type(subtask.task_type)
        affordable = [s for s in candidates if s.cost_per_call <= remaining_budget]
        if not affordable:
            return SelectionResult(
                SelectionStatus.INSUFFICIENT_BUDGET, None,
                f"预算不足：{subtask.task_type} 类别无可支付服务",
            )
        cheapest = min(affordable, key=lambda s: s.cost_per_call)
        return SelectionResult(
            SelectionStatus.OK, cheapest,
            f"选最便宜的可支付服务 {cheapest.name}",
        )


class _HardStopSelector:
    """Budget Hard-Stop Agent 用：满足质量且可支付则选质量最高者，
    否则直接停止（无降级逻辑）。"""

    def __init__(self, registry: ServiceRegistry) -> None:
        self._registry = registry

    def select(self, subtask: Subtask, remaining_budget: float) -> SelectionResult:
        candidates = self._registry.list_by_type(subtask.task_type)
        qualified_affordable = [
            s for s in candidates
            if s.quality_score >= subtask.required_min_quality
            and s.cost_per_call <= remaining_budget
        ]
        if not qualified_affordable:
            return SelectionResult(
                SelectionStatus.INSUFFICIENT_BUDGET, None,
                f"预算无法按质量要求执行下一步（无降级逻辑），硬停止",
            )
        best = max(qualified_affordable, key=lambda s: s.quality_score)
        return SelectionResult(
            SelectionStatus.OK, best,
            f"选满足质量要求且可支付的质量最高者 {best.name}",
        )


class _NoReplanner:
    """无重规划能力的重规划器：任何失败/质量不足直接 ABORT。"""

    @staticmethod
    def should_trigger(subtask: Subtask, result: ServiceResult) -> bool:
        return (not result.success) or (
            result.quality < subtask.required_min_quality
        )

    def replan(
        self,
        current: Subtask,
        result: ServiceResult,
        remaining_subtasks: list[Subtask],
        remaining_budget: float,
    ) -> ReplanResult:
        return ReplanResult(
            ReplanAction.ABORT, [],
            "基线无重规划能力，直接终止",
        )


# ---------- 指标 ----------

@dataclass(frozen=True)
class TaskRunMetrics:
    """单任务单智能体单预算档位的一次运行指标。"""

    task_id: str
    agent: str
    budget_tier: str
    budget: float             # 参照预算（该档位额度）
    success: bool             # 状态为 completed 记为成功
    spent: float              # 实际模拟开销
    over_budget: bool         # 是否超出参照预算
    tool_calls: int           # 实际执行的服务调用次数
    repeated_calls: int       # 冗余重复调用（重试/浪费的失败调用）
    completeness: float       # LLM-Judge 完整性
    correctness: float        # LLM-Judge 事实正确性（模拟代理）
    judge_overall: float      # LLM-Judge 综合分

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 智能体装配 ----------

def build_agent(name: str, seed: int | None = None) -> MultiAgentOrchestrator:
    """按名称装配基线/完整版智能体（proposal 3.4）。

    所有方案统一走多 agent 编排（调研 / 询价 / 执行 / 审计），
    差异只在决策策略：完整版用预算感知选择 + 动态重规划，
    基线分别用质量优先、最低价、硬停止且不做重规划。
    """
    registry = ServiceRegistry()
    if name == "standard":
        return build_enterprise_multi_agent(
            seed=seed or 0, registry=registry,
            selector=_QualityFirstSelector(registry), replanner=_NoReplanner(),
        )
    if name == "min_cost":
        return build_enterprise_multi_agent(
            seed=seed or 0, registry=registry,
            selector=_MinCostSelector(registry), replanner=_NoReplanner(),
        )
    if name == "hard_stop":
        return build_enterprise_multi_agent(
            seed=seed or 0, registry=registry,
            selector=_HardStopSelector(registry), replanner=_NoReplanner(),
        )
    if name == "proposed":
        return build_enterprise_multi_agent(seed=seed or 0, registry=registry)
    raise ValueError(f"未知智能体: {name}（可选: {AGENT_NAMES}）")


def build_ablation_agent(name: str, seed: int | None = None) -> MultiAgentOrchestrator:
    """按名称装配消融变体（proposal 3.6），同样基于多 agent 编排。"""
    registry = ServiceRegistry()
    if name == "full":
        return build_enterprise_multi_agent(seed=seed or 0, registry=registry)
    if name == "no_budget_aware_selection":
        # 禁用预算感知选择：无降级，质量优先，付不起即停
        return build_enterprise_multi_agent(
            seed=seed or 0, registry=registry,
            selector=_HardStopSelector(registry),
        )
    if name == "no_dynamic_replanning":
        # 禁用动态重规划：任何失败/质量不足直接终止
        return build_enterprise_multi_agent(
            seed=seed or 0, registry=registry,
            selector=BudgetAwareSelector(registry), replanner=_NoReplanner(),
        )
    raise ValueError(f"未知消融变体: {name}（可选: {ABLATION_NAMES}）")


# ---------- 评估器 ----------

def load_tasks(path: str | Path = _DEFAULT_TASKS_PATH) -> tuple[list[dict], dict]:
    """加载测试任务集与预算档位。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["tasks"], data["budget_tiers"]


class Evaluator:
    """评估器：运行任务集并汇总指标。"""

    def __init__(self, judge: LLMJudge | None = None) -> None:
        self._planner = TaskPlanner()
        self._judge = judge if judge is not None else LLMJudge()

    def run_single(
        self,
        agent: MultiAgentOrchestrator,
        agent_name: str,
        task_entry: dict,
        tier_name: str,
        tier_budget: float,
    ) -> TaskRunMetrics:
        """运行单个任务并计算指标。"""
        internal_budget = (
            STANDARD_INTERNAL_BUDGET if agent_name == "standard" else tier_budget
        )
        report: RunReport = agent.run(task_entry["task"], internal_budget)
        planned_count = len(self._planner.decompose(task_entry["task"]))
        judge = self._judge.score(task_entry["task"], report.outputs, planned_count)

        tool_calls = sum(1 for s in report.steps if s.executed)
        return TaskRunMetrics(
            task_id=task_entry["id"],
            agent=agent_name,
            budget_tier=tier_name,
            budget=tier_budget,
            success=report.state is AgentState.COMPLETED,
            spent=report.spent,
            over_budget=report.spent > tier_budget,
            tool_calls=tool_calls,
            repeated_calls=tool_calls - len(report.outputs),
            completeness=judge.completeness,
            correctness=judge.correctness,
            judge_overall=judge.overall,
        )

    def run_suite(
        self,
        agent_names: tuple[str, ...] = AGENT_NAMES,
        tasks: list[dict] | None = None,
        tiers: dict | None = None,
        seed_base: int = 1000,
        ablation: bool = False,
    ) -> list[TaskRunMetrics]:
        """批量运行：智能体 x 预算档位 x 任务。

        同一任务下各智能体使用相同 seed，保证失败模式一致、对比公平。
        ablation=True 时用 build_ablation_agent 装配。
        """
        if tasks is None or tiers is None:
            tasks, tiers = load_tasks()
        build = build_ablation_agent if ablation else build_agent
        rows: list[TaskRunMetrics] = []
        for tier_name, tier_budget in tiers.items():
            for agent_name in agent_names:
                for i, task_entry in enumerate(tasks):
                    agent = build(agent_name, seed=seed_base + i)
                    rows.append(self.run_single(
                        agent, agent_name, task_entry, tier_name, tier_budget,
                    ))
        return rows

    @staticmethod
    def aggregate(rows: list[TaskRunMetrics]) -> dict:
        """汇总一组运行指标（proposal 3.5 口径）。"""
        n = max(1, len(rows))
        return {
            "runs": len(rows),
            "success_rate": round(sum(r.success for r in rows) / n, 4),
            "avg_spent": round(sum(r.spent for r in rows) / n, 4),
            "over_budget_ratio": round(sum(r.over_budget for r in rows) / n, 4),
            "avg_tool_calls": round(sum(r.tool_calls for r in rows) / n, 2),
            "avg_repeated_calls": round(sum(r.repeated_calls for r in rows) / n, 2),
            "avg_completeness": round(sum(r.completeness for r in rows) / n, 4),
            "avg_correctness": round(sum(r.correctness for r in rows) / n, 4),
            "avg_judge_overall": round(sum(r.judge_overall for r in rows) / n, 4),
        }
