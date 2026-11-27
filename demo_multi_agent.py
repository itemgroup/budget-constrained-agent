"""Multi-agent layer demo (star topology + shared blackboard).

多智能体层演示：
- 案例 1：企业任务 —— 多 agent 与单 agent 同参数对照（决策应完全一致）；
- 案例 2：行程规划 —— 复用同一套编排，验证场景泛化；
- 案例 3：图记忆 —— 同一任务跑两次，第二次调研阶段命中记忆、分解零 LLM。

运行方式（项目根目录下）：
    python demo_multi_agent.py
"""

from __future__ import annotations

import time
from pathlib import Path

from src.agent_core import AgentOrchestrator
from src.memory import ExperienceGraph
from src.multi_agent import (
    MultiAgentRunReport,
    build_enterprise_multi_agent,
    build_travel_multi_agent,
)
from src.simulated_services import SimulatedServiceRunner
from src.task_planner import TaskPlanner

LINE = "=" * 64


def _print_agent_log(report: MultiAgentRunReport) -> None:
    print("\n── 黑板活动日志（协作过程）──")
    for m in report.messages:
        cost = f"（计费 ${m.cost:.3f}）" if m.cost else ""
        print(f"  {m.agent:<24}{m.action:<10}{m.detail}{cost}")


def _print_audit(report: MultiAgentRunReport) -> None:
    audit = report.audit_report
    print("\n── 审计报告 ──")
    if audit is None:
        print("  （无审计结论）")
        return
    print(f"  通过    : {'是' if audit.passed else '否'}")
    print(f"  对账    : {audit.reconciliation_detail}")
    print(f"  格式    : {'合规' if not audit.format_errors else audit.format_errors}")
    if audit.score is not None:
        s = audit.score
        print(
            f"  评分    : completeness={s.completeness:.2f} "
            f"correctness={s.correctness:.2f} overall={s.overall:.2f}"
        )
        print(f"  评分说明: {s.rationale}")
    print(f"  结论    : {audit.reason}")


def case_enterprise() -> None:
    print(LINE)
    print("案例 1 · 企业任务（$0.50 · seed 5）—— 多 agent vs 单 agent 对照")
    print(LINE)
    task = "调研行业数据，分析市场机会并生成季度业务报告"
    budget, seed = 0.50, 5

    single = AgentOrchestrator(
        planner=TaskPlanner(), runner=SimulatedServiceRunner(seed=seed)
    ).run(task, budget)

    agent = build_enterprise_multi_agent(seed=seed)
    multi = agent.run(task, budget)

    print(f"\n  [单 agent] 状态 {single.state.value} · 消耗 ${single.spent:.2f} "
          f"· 轮次 {len(single.steps)} · 产出 {len(single.outputs)}")
    print(f"  [多 agent] 状态 {multi.state.value} · 消耗 ${multi.spent:.2f} "
          f"· 轮次 {len(multi.steps)} · 产出 {len(multi.outputs)}")

    same_services = [s.service_name for s in single.steps] == [
        s.service_name for s in multi.steps
    ]
    same_money = abs(single.spent - multi.spent) < 1e-9
    same_state = single.state is multi.state
    print(f"\n  决策一致性: 服务序列 {'✓' if same_services else '✗'} · "
          f"开销 {'✓' if same_money else '✗'} · 状态 {'✓' if same_state else '✗'}")

    _print_agent_log(multi)
    _print_audit(multi)

    print("\n── 合规交付物（节选）──")
    deliverable = multi.audit_report.deliverable if multi.audit_report else ""
    print("\n".join(deliverable.splitlines()[:9]))

    print("\n── 账本 ──")
    print(multi.ledger_summary)


def case_travel() -> None:
    print("\n" + LINE)
    print("案例 2 · 行程规划（$800 · seed 42）—— 场景泛化验证")
    print(LINE)
    task = "安排一趟 3 天的东京行程，体验优先"
    budget, seed = 800.0, 42

    agent = build_travel_multi_agent(seed=seed)
    report = agent.run(task, budget)

    print(f"\n  状态 {report.state.value} · 消耗 ${report.spent:.2f} / ${budget:.2f} "
          f"· 轮次 {len(report.steps)} · 确认预订 {len(report.outputs)}")
    print(f"  报价单: {len(agent.blackboard.price_sheet)} 项")

    _print_agent_log(report)
    _print_audit(report)

    print("\n── 预订确认 ──")
    for out in report.outputs:
        print(f"  · [{out.task_type}] {out.subtask_desc} → {out.service_name}"
              f"（质量 {out.quality:.2f}）")


def case_memory() -> None:
    print("\n" + LINE)
    print("案例 3 · 图记忆（同一任务跑两次：第二次调研命中，分解零 LLM）")
    print(LINE)
    task = "调研行业数据，分析市场机会并生成季度业务报告"
    budget, seed = 0.50, 5

    # 每次进程用独立的记忆文件，保证「第一次未命中、第二次命中」可复现
    mem_path = Path("data") / f"multi_agent_demo_{time.time_ns()}.json"
    memory = ExperienceGraph(path=mem_path)

    print("\n  第 1 次运行（冷启动）")
    first = build_enterprise_multi_agent(seed=seed, memory=memory).run(task, budget)
    print(f"    记忆命中 {first.research_notes.get('hit')} · "
          f"消耗 ${first.spent:.2f} · 产出 {len(first.outputs)}")
    for m in first.messages:
        if m.agent in ("ResearchAgent", "MultiAgentOrchestrator"):
            print(f"    {m.agent:<24}{m.detail}")

    print("\n  第 2 次运行（热启动）")
    second = build_enterprise_multi_agent(seed=seed, memory=memory).run(task, budget)
    notes = second.research_notes
    print(f"    记忆命中 {notes.get('hit')}（{'精确' if notes.get('exact') else '相似'}，"
          f"相似度 {notes.get('similarity')}）· 消耗 ${second.spent:.2f} "
          f"· 产出 {len(second.outputs)}")
    for m in second.messages:
        if m.agent in ("ResearchAgent", "MultiAgentOrchestrator"):
            print(f"    {m.agent:<24}{m.detail}")

    print(f"\n  经验图规模: {memory.stats()['nodes']} · "
          f"累计命中 {memory.stats()['total_hits']}")
    print(f"  记忆文件: {mem_path}（data/ 已 gitignore）")


if __name__ == "__main__":
    case_enterprise()
    case_travel()
    case_memory()
    print("\n" + LINE)
    print("演示结束")
    print(LINE)
