"""Memory-layer demo.

图结构长期记忆演示：验证「类似任务复用分解经验、少调 LLM 省 token」。

演示逻辑：
1. planner1 分解任务 A → 记忆未命中 → 产生分解并写入经验图；
2. planner2（全新实例，共享同一个记忆文件）分解任务 A'（与 A 措辞相近）
   → 相似命中 → 复用 A 的分解链，LLM 调用次数 0；
3. planner2 分解完全不同的任务 B → 未命中 → 正常分解并入库；
4. 运行完整智能体后回填服务使用统计（step -served_by-> service），
   展示图的"经验越攒越准"；
5. 打印图的规模、命中计数与 token 节省报告。

第二次运行本脚本时，第 1/3 步也会直接命中磁盘记忆——持久化生效。

运行方式（项目根目录下）：
    .\\budget_agent_env\\Scripts\\python.exe demos\\demo_memory.py --fresh   # 清空记忆冷启动
    .\\budget_agent_env\\Scripts\\python.exe demos\\demo_memory.py           # 带历史记忆运行
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_core import AgentOrchestrator
from src.llm_task_planner import LLMTaskPlanner
from src.memory import DEFAULT_MEMORY_PATH, similarity
from src.simulated_services import SimulatedServiceRunner

# 三个演示任务：A 与 A' 措辞相近（应命中），B 完全不同（应未命中）
TASK_A = "调研行业数据，分析市场机会并生成季度业务报告"
TASK_A2 = "调研行业数据，分析市场格局并生成季度业务报告"
TASK_B = "整理三份供应商合同，解析关键条款并生成法律风险摘要"


def show_decomposition(planner: LLMTaskPlanner, tag: str, task: str) -> None:
    """分解并打印子任务序列与计数器。"""
    subtasks = planner.decompose(task)
    print(f"\n[{tag}] 任务: {task}")
    for st in subtasks:
        print(f"    {st.order}. [{st.task_type}] {st.desc} "
              f"(质量要求 {st.required_min_quality:.2f}, 预估 ${st.estimated_cost:.2f})")
    print(f"    → 累计: LLM 调用 {planner.llm_calls} 次, 记忆命中 {planner.memory_hits} 次")


def main() -> None:
    fresh = "--fresh" in sys.argv
    if fresh:
        DEFAULT_MEMORY_PATH.unlink(missing_ok=True)
    print("=" * 72)
    print("图结构长期记忆演示（相似任务复用分解经验，节省 LLM token）")
    print(f"（{'--fresh 冷启动：记忆已清空' if fresh else '带历史记忆运行（持久化生效）'}）")
    print("=" * 72)
    print(f"\n词面相似度自检: A vs A' = {similarity(TASK_A, TASK_A2):.2f}"
          f"（应较高）; A vs B = {similarity(TASK_A, TASK_B):.2f}（应较低）")

    # 1) 全新 planner 分解任务 A：第一次见 → 未命中 → 入库
    p1 = LLMTaskPlanner()
    show_decomposition(p1, "planner1 · 任务A（首次）", TASK_A)

    # 2) 全新 planner + 措辞相近的任务 A'：应相似命中（跨实例复用）
    p2 = LLMTaskPlanner()
    show_decomposition(p2, "planner2 · 任务A'（相似）", TASK_A2)

    # 3) 完全不同的任务 B：未命中 → 正常分解入库
    show_decomposition(p2, "planner2 · 任务B（首次）", TASK_B)

    # 4) 回填一次真实运行的服务使用统计，图的 service_hints 越用越准
    agent = AgentOrchestrator(
        planner=p2, runner=SimulatedServiceRunner(seed=3))
    report = agent.run(TASK_A, 0.5)
    p2.memory.record_run_outputs(TASK_A, report.steps)
    print(f"\n[运行回填] 任务A 实际执行 {len(report.outputs)} 个子任务，"
          f"服务使用统计已写入经验图")

    # 5) 图的规模与命中报告
    s = p2.memory.stats()
    print(f"\n——— 经验图规模 ———")
    print(f"    节点: {s['nodes']}   边: {s['edges']}   累计命中: {s['total_hits']}")
    print(f"    持久化文件: {s['file']}")
    rows = p2.memory.service_stats()
    if rows:
        print(f"\n——— 服务经验统计（step -served_by-> service）———")
        for r in rows:
            print(f"    [{r['task_type']:<16}] {r['service']:<16} "
                  f"{r['calls']} 次 · 平均质量 {r['avg_quality']:.2f} · 平均 ${r['avg_cost']:.2f}")

    print(f"\n——— token 节省报告 ———")
    print(f"    planner1: LLM 调用 {p1.llm_calls} 次, 记忆命中 {p1.memory_hits} 次")
    print(f"    planner2: LLM 调用 {p2.llm_calls} 次, 记忆命中 {p2.memory_hits} 次")
    print(f"    → 同类任务再次出现时，分解阶段 LLM 调用可降为 0")

    print("\n" + "=" * 72)
    print("观察要点：")
    print("  1) planner2 是全新实例，却能命中 planner1 的经验——记忆在图里，不在实例里；")
    print("  2) 再次运行本脚本，任务A 首次分解就会命中磁盘记忆——持久化生效；")
    print("  3) 经验图含服务统计（哪类步骤历史上由哪个服务做得好），是多 agent 黑板的地基。")


if __name__ == "__main__":
    main()
