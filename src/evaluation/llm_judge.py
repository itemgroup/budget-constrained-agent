"""LLM judge for output quality scoring.

LLM-Judge：评估智能体最终文本输出的【完整性】与【事实正确性】（proposal 3.5）。

当前为模拟评分器（启发式，不调用真实 LLM）：
- completeness：完成子任务数 / 计划子任务数（SKIP/未执行会扣完整性）；
- correctness：以服务 quality_score 的均值作为事实正确性的模拟代理。

真实 LLM 评分器 RealLLMJudge（LangChain + DeepSeek）见本文件末尾，
与 LLMJudge 接口一致，可互换注入 Evaluator / main.py；
无 DEEPSEEK_API_KEY 或调用失败时回退到模拟评分（fallback=True）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

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


#: 评分提示词：要求严格输出 JSON 对象（0~1 分制，与 JudgeScore 一致）
_JUDGE_PROMPT = """你是输出质量评审。请评估以下企业任务的执行结果。

原始任务：{task}
计划子任务数：{planned}
实际完成子任务数：{done}
各子任务输出：
{outputs}

从两个维度打分（0~1 的小数）：
- completeness：输出对任务需求的覆盖完整程度
- correctness：输出内容的事实正确性与合理性

只输出 JSON 对象，不要输出任何其他文字：
{{"completeness": 0.0, "correctness": 0.0, "rationale": "一句话评分理由"}}"""


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


class RealLLMJudge:
    """真实 LLM 评分器（LangChain + DeepSeek），接口与 LLMJudge 一致。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        fallback: bool = True,
    ) -> None:
        load_dotenv()
        self.fallback = fallback
        self._chain = None
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            if not fallback:
                raise RuntimeError("未找到 DEEPSEEK_API_KEY（请在 .env 中配置）")
            return

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=temperature,
        )
        prompt = ChatPromptTemplate.from_template(_JUDGE_PROMPT)
        self._chain = prompt | llm | StrOutputParser()

    def score(
        self,
        task: str,
        outputs: list[SubtaskOutput],
        planned_count: int,
    ) -> JudgeScore:
        """对一次运行的全部子任务输出打分（签名与 LLMJudge.score 一致）。"""
        if self._chain is None or not outputs:
            # 无 key 或无产出：直接使用模拟评分
            return LLMJudge().score(task, outputs, planned_count)
        text = "\n".join(
            f"- [{o.task_type}] {o.subtask_desc}: {o.output}" for o in outputs
        )
        try:
            raw = self._chain.invoke({
                "task": task,
                "outputs": text,
                "planned": planned_count,
                "done": len(outputs),
            })
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("LLM 输出中不含 JSON 对象")
            data = json.loads(raw[start:end + 1])
            comp = _clamp01(float(data["completeness"]))
            corr = _clamp01(float(data["correctness"]))
            overall = round(0.5 * comp + 0.5 * corr, 4)
            return JudgeScore(comp, corr, overall,
                              str(data.get("rationale", "")))
        except Exception as e:
            if not self.fallback:
                raise
            s = LLMJudge().score(task, outputs, planned_count)
            return JudgeScore(
                s.completeness, s.correctness, s.overall,
                f"[LLM 评分失败回退] {s.rationale}（{e}）",
            )
