"""ResearchAgent: consult the experience graph before task decomposition.

调研员 Agent：负责在分解前查询图记忆，判断是否有相似历史任务可以直接复用。

- 命中（精确 / 相似 >= 阈值）：把历史分解链与 service_hints 写入黑板，
  主 Agent 直接复用，**分解阶段零 LLM 调用**（省 token）；
- 未命中：标记 hit=False，主 Agent 走正常分解流程。

模拟环境的全部调研能力就是图记忆（离线、零成本）；
真实环境在此处扩展 RAG 私有知识库与外部搜索即可——接口不变。
"""

from __future__ import annotations

from ..memory import DEFAULT_SIMILARITY_THRESHOLD, ExperienceGraph
from .base import Agent


class ResearchAgent(Agent):
    """调研员：查询经验记忆，产出 research_notes。"""

    name = "ResearchAgent"

    def __init__(self, memory: ExperienceGraph | None = None) -> None:
        self._memory = memory

    def act(self, blackboard) -> None:
        """查记忆并把结论写入黑板。"""
        if self._memory is None:
            blackboard.research_notes = {
                "hit": False,
                "reason": "未接入经验记忆，走标准分解流程",
            }
            self._log(blackboard, "lookup", "未接入经验记忆，跳过检索")
            return

        hit = self._memory.lookup(blackboard.task)
        if hit is None:
            blackboard.research_notes = {
                "hit": False,
                "reason": f"无相似度 >= {DEFAULT_SIMILARITY_THRESHOLD} 的历史任务",
            }
            self._log(
                blackboard, "lookup",
                f"未命中（阈值 {DEFAULT_SIMILARITY_THRESHOLD}），需重新分解",
            )
            return

        blackboard.research_notes = {
            "hit": True,
            "exact": hit.exact,
            "similarity": round(hit.similarity, 4),
            "source_task": hit.source_task,
            "plan": list(hit.subtasks),
            "service_hints": hit.service_hints,
        }
        kind = "精确命中" if hit.exact else f"相似命中（相似度 {hit.similarity:.2f}）"
        hint_types = "、".join(hit.service_hints) if hit.service_hints else "暂无服务提示"
        self._log(
            blackboard, "lookup",
            f"{kind}：复用「{hit.source_task}」的 {len(hit.subtasks)} 步分解链，"
            f"分解阶段零 LLM 调用；服务提示：{hint_types}",
        )
