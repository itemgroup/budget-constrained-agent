"""Travel-scenario extension demo.

旅行场景扩展包演示：同一引擎（分解 -> 预算感知选选项 -> 预订 -> 重规划）
安排整趟行程，展示三种预算档位下的不同行为：

- 宽裕 $1500：尽量选高端选项，偶有降级；
- 标准 $800：明显降级（商务舱 -> 经济舱、五星 -> 三星），行程完整；
- 紧张 $400：极端降级 + 环节被砍，甚至预算耗尽终止。

每次运行按 seed 生成"本次查询到的报价单"（基准价 ±15% 波动），
先打印报价单再执行，模拟"agent 自主查询价格后再安排"。

运行方式（项目根目录下）：
    .\\budget_agent_env\\Scripts\\python.exe demos\\demo_travel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_core import AgentState
from src.scenarios.scenario import (
    build_price_registry,
    load_catalog,
    build_travel_agent,
)

# 演示案例：名称 / 任务文本 / 总预算 / 种子
CASES = [
    ("宽裕预算", "安排一趟 3 天的东京行程，体验优先", 1500.0, 11),
    ("标准预算", "安排一趟 3 天的东京行程，兼顾体验与花费", 800.0, 42),
    ("紧张预算", "安排一趟 3 天的东京行程，穷游", 400.0, 7),
    ("紧张预算·换个报价", "安排一趟 3 天的东京行程，穷游", 400.0, 8),
]

STATE_ZH = {
    AgentState.COMPLETED: "行程安排完成",
    AgentState.BUDGET_EXHAUSTED: "预算耗尽，行程终止",
    AgentState.FAILED: "行程异常终止",
    AgentState.RUNNING: "进行中",
    AgentState.IDLE: "未开始",
}


def print_price_sheet(registry, types: tuple[str, ...]) -> None:
    """打印本次查询到的报价单（按环节类别分组）。"""
    print("  ——— 本次查询到的报价单（基准价 ±15% 波动） ———")
    for t in types:
        opts = sorted(
            registry.list_by_type(t), key=lambda s: -s.cost_per_call
        )
        for s in opts:
            print(f"    [{t:<8}] {s.name:<18} ${s.cost_per_call:>7.2f}  体验 {s.quality_score:.2f}")


def print_report(state, reason, steps, outputs, skipped, spent, remaining, budget) -> None:
    """打印一次运行的行程结果。"""
    print(f"  状态: {STATE_ZH.get(state, state)}   终止原因: {reason}")
    print(f"  花费 ${spent:.2f} / 预算 ${budget:.2f}   剩余 ${remaining:.2f}")
    print("  行程执行轨迹:")
    for s in steps:
        if not s.executed:
            print(f"    轮次 {s.step:>2} | {s.subtask_desc} —— 未执行（{s.note}）")
            continue
        flag = {"ok": "正常", "downgraded": "降级", "insufficient_budget": "预算不足"}.get(
            s.selection_status, s.selection_status)
        succ = "" if s.success else "  [预订失败]"
        print(f"    轮次 {s.step:>2} | {s.subtask_desc}: {s.service_name}"
              f" ${s.cost:.2f} 体验{s.quality:.2f} [{flag}]{succ}")
        if s.note:
            print(f"          └─ {s.note}")
    print("  预订确认:")
    for o in outputs:
        print(f"    · {o.output}")
    if skipped:
        print(f"  被砍掉的环节: {'、'.join(skipped)}")
    if not outputs and not skipped:
        print("    （无成功预订）")


def main() -> None:
    catalog = load_catalog()
    types = tuple(dict.fromkeys(step.task_type for step in catalog.plan))
    print("=" * 72)
    print(f"旅行场景扩展包演示 · 目录「{catalog.name}」{catalog.description}")
    print(f"目录选项 {len(catalog.options)} 个，行程模板 "
          f"{' -> '.join(f'{s.task_type}x{s.count}' for s in catalog.plan)}")
    print("=" * 72)

    for name, task, budget, seed in CASES:
        print(f"\n【{name}】预算 ${budget:.2f} · seed={seed}")
        print(f"  任务: {task}")

        agent, registry = build_travel_agent(seed=seed)
        print_price_sheet(registry, types)

        report = agent.run(task, budget)
        print_report(
            report.state, report.termination_reason, report.steps,
            report.outputs, report.skipped_subtasks,
            report.spent, report.remaining, report.total_budget,
        )

    print("\n" + "=" * 72)
    print("演示结束。观察要点：")
    print("  1) 同一引擎零修改复用：选项即服务，预订即调用，账本/降级/重规划全部原样工作；")
    print("  2) 预算越紧，自动降级越明显（商务舱->经济舱、五星->青旅、餐厅->街头小吃）；")
    print("  3) 售罄/满房触发重规划：RETRY 换次优选项、SKIP 砍环节、预算耗尽体面终止；")
    print("  4) 不同 seed 报价单不同，穷游预算可能时而完不成——价格波动带来的真实风险。")


if __name__ == "__main__":
    main()
