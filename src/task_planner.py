"""Task decomposition (simulated version).

任务分解器：将自然语言企业任务拆成有序子任务列表。

当前为模拟版：基于关键词的硬编码分解逻辑，不接入 LLM。
扩展接口：后续 LLM 驱动的分解器只需实现相同的方法签名
    decompose(task: str) -> list[Subtask]
即可无缝替换本实现，上层调用方无需改动。

与其他模块解耦：本模块不依赖 BudgetLedger / ServiceRegistry，
只输出 Subtask 数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Subtask:
    """一个有序子任务。"""

    order: int                  # 执行顺序，从 1 开始
    task_type: str              # 子任务类型，对应服务类别：Search / Document Parsing / LLM Reasoning / Report Generation
    required_min_quality: float # 该子任务可接受的最低服务质量分（0~1）
    estimated_cost: float       # 预估开销（美元），用于计划级预算预估
    desc: str                   # 子任务描述


# 各任务类型的配置：(required_min_quality, estimated_cost, 触发关键词)
# required_min_quality / estimated_cost 为模拟版硬编码值，
# estimated_cost 取该类别最低档服务价格作为乐观基线（proposal 3.2）。
_TYPE_PROFILES: dict[str, tuple[float, float, tuple[str, ...]]] = {
    "Search":            (0.70, 0.01, ("调研", "检索", "搜索", "research", "search")),
    "Document Parsing":  (0.60, 0.02, ("文档解析", "解析", "文档", "parse", "document")),
    "LLM Reasoning":     (0.75, 0.10, ("分析", "推理", "analyze", "reason")),
    "Report Generation": (0.65, 0.05, ("报告", "摘要", "report", "summary")),
}

# 固定流水线顺序：检索 -> 解析 -> 推理 -> 报告
_PIPELINE_ORDER: tuple[str, ...] = (
    "Search",
    "Document Parsing",
    "LLM Reasoning",
    "Report Generation",
)

# 各任务类型的默认描述（模拟版）
_TYPE_DESC: dict[str, str] = {
    "Search": "检索相关资料",
    "Document Parsing": "解析文档内容",
    "LLM Reasoning": "分析并推理关键信息",
    "Report Generation": "生成摘要报告",
}


class TaskPlanner:
    """任务分解器（模拟版）：按关键词匹配，按固定流水线顺序输出子任务。"""

    def decompose(self, task: str) -> list[Subtask]:
        """将自然语言企业任务拆分为有序子任务列表。

        规则：任务文本中出现某类型关键词即纳入该类型子任务，
        按 _PIPELINE_ORDER 固定顺序排列；一个关键词都没匹配到时，
        返回完整默认流水线（保证可运行）。
        """
        text = task.lower()
        matched = [
            t for t in _PIPELINE_ORDER
            if any(kw.lower() in text for kw in _TYPE_PROFILES[t][2])
        ]
        if not matched:
            matched = list(_PIPELINE_ORDER)

        subtasks: list[Subtask] = []
        for i, task_type in enumerate(matched, start=1):
            min_quality, est_cost, _ = _TYPE_PROFILES[task_type]
            subtasks.append(
                Subtask(
                    order=i,
                    task_type=task_type,
                    required_min_quality=min_quality,
                    estimated_cost=est_cost,
                    desc=_TYPE_DESC[task_type],
                )
            )
        return subtasks
