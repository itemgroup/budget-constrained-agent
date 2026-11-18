"""LLM judge for output quality scoring.

LLM-Judge：评估智能体最终文本输出的【完整性】与【事实正确性】（proposal 3.5）。

当前为模拟评分器（启发式，不调用真实 LLM）：
- completeness：完成子任务数 / 计划子任务数（SKIP/未执行会扣完整性）；
- correctness：以服务 quality_score 的均值作为事实正确性的模拟代理。

扩展接口：后续以 LangChain 接入真实 LLM 时，只需实现相同签名
    score(task: str, outputs: list[SubtaskOutput], planned_count: int) -> JudgeScore
并注入 Evaluator 即可无缝替换。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent_core import SubtaskOutput


@dataclass(frozen=True)
class JudgeScore:
    """LLM-Judge 评分结果（0~1）。"""

    completeness: float  # 完整性：子任务覆盖度
    correctness: float   # 事实正确性（模拟版为质量分代理）
    overall: float       # 综合分（两项等权）
    rationale: str       # 评分说明


class LLMJudge:
    """LLM-Judge 评分器（当前为模拟实现）。"""

    def score(
        self,
        task: str,
        outputs: list[SubtaskOutput],
        planned_count: int,
    ) -> JudgeScore:
        """对一次运行的全部子任务输出打分。

        参数:
            task: 原始任务文本
            outputs: 成功完成的子任务输出列表
            planned_count: 任务分解得到的计划子任务总数
        """
        planned = max(1, planned_count)
        completeness = round(min(1.0, len(outputs) / planned), 4)
        correctness = (
            round(sum(o.quality for o in outputs) / len(outputs), 4)
            if outputs
            else 0.0
        )
        overall = round(0.5 * completeness + 0.5 * correctness, 4)
        rationale = (
            f"[模拟评分] 完成 {len(outputs)}/{planned_count} 个子任务 "
            f"(completeness={completeness:.2f})；"
            f"平均产出质量 {correctness:.2f} (correctness 代理)"
        )
        return JudgeScore(completeness, correctness, overall, rationale)
