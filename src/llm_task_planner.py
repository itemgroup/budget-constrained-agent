"""LLM-driven task planner (LangChain + DeepSeek).

真实 LLM 版任务分解器，接口与 TaskPlanner 完全一致：
    decompose(task: str) -> list[Subtask]
可注入 AgentOrchestrator(planner=...) 无缝替换模拟版（关键词匹配）。

设计要点：
- LLM 只负责"理解与分解"，输出 task_type / required_min_quality / desc；
- estimated_cost 不信任 LLM，按类别最低服务价从注册表确定性填充；
- 两级缓存省 token：实例内精确缓存（同任务字符串只调一次 LLM）+ 图结构
  长期记忆（src/memory.py，跨实例/跨运行的精确与相似命中均跳过 LLM 调用，
  并把每次分解经验持久化到 data/experience_graph.json）；
- 无 DEEPSEEK_API_KEY 或 LLM 调用失败时回退到模拟版 TaskPlanner
  （fallback=True，默认开启；置 False 则直接抛异常）。

token 计数：llm_calls（实际发起的 LLM 调用次数）、memory_hits（记忆命中
省下的调用次数）——demo/前端可据此展示"记忆省了多少 token"。
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from .memory import ExperienceGraph
from .service_registry import ServiceRegistry
from .task_planner import Subtask, TaskPlanner

#: 允许的任务类型（与 Service.service_type 对应）
ALLOWED_TYPES = ("Search", "Document Parsing", "LLM Reasoning", "Report Generation")

#: 分解提示词：要求严格输出 JSON 数组
PROMPT_TEMPLATE = """你是一个企业任务分解器。把下面的企业任务分解为有序的子任务序列。

子任务类型 task_type 只能取以下之一：
- "Search"：信息检索
- "Document Parsing"：文档解析
- "LLM Reasoning"：分析、推理、摘要
- "Report Generation"：报告/文档生成

每个子任务包含三个字段：
- task_type：上述类型之一
- required_min_quality：0~1 的小数，该子任务可接受的最低服务质量
- desc：一句话子任务描述

要求：按执行顺序排列；只输出 JSON 数组，不要输出任何其他文字。
示例：[{{"task_type": "Search", "required_min_quality": 0.7, "desc": "检索行业公开数据"}}]

任务：{task}"""


class LLMTaskPlanner:
    """基于 LangChain + DeepSeek 的任务分解器（与 TaskPlanner 接口一致）。"""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        fallback: bool = True,
        memory: ExperienceGraph | None = None,
    ) -> None:
        load_dotenv()
        self.registry = registry if registry is not None else ServiceRegistry()
        self.fallback = fallback
        self._simulated = TaskPlanner()
        self._cache: dict[str, list[Subtask]] = {}
        self._chain = None
        # 图结构长期记忆：跨实例/跨运行复用分解经验，命中即省一次 LLM 调用
        self.memory = memory if memory is not None else ExperienceGraph()
        self.llm_calls = 0      # 实际发起的 LLM 调用次数
        self.memory_hits = 0    # 记忆命中（省掉的 LLM 调用）次数

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            msg = "未找到 DEEPSEEK_API_KEY（请在 .env 中配置）"
            if not fallback:
                raise RuntimeError(msg)
            print(f"[LLMTaskPlanner] {msg}，使用模拟分解器")
            return

        # 延迟导入：未使用真实 LLM 时不要求 langchain 可用
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=temperature,
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self._chain = prompt | llm | StrOutputParser()

    def decompose(self, task: str) -> list[Subtask]:
        """把自然语言任务分解为有序子任务列表（实例缓存 + 图记忆两级复用）。"""
        if task in self._cache:
            return list(self._cache[task])

        # 1) 图记忆：精确/相似命中均直接复用历史分解，跳过 LLM 调用
        hit = self.memory.lookup(task)
        if hit is not None:
            self.memory_hits += 1
            kind = "精确" if hit.exact else f"相似 {hit.similarity:.2f}"
            print(f"[Memory] {kind}命中历史经验（源任务: {hit.source_task}），"
                  f"复用 {len(hit.subtasks)} 个子任务，节省一次 LLM 调用")
            self._cache[task] = hit.subtasks
            return list(hit.subtasks)

        # 2) 无 key：模拟分解器，结果同样写入记忆供下次复用
        if self._chain is None:
            subtasks = self._simulated.decompose(task)
            self.memory.add_experience(task, subtasks)
            self._cache[task] = subtasks
            return list(subtasks)

        # 3) 真实 LLM 分解；失败回退模拟（回退结果不入记忆，避免污染经验库）
        self.llm_calls += 1
        try:
            raw = self._chain.invoke({"task": task})
            subtasks = self._parse(raw)
        except Exception as e:
            if not self.fallback:
                raise
            print(f"[LLMTaskPlanner] LLM 分解失败（{e}），回退到模拟分解器")
            subtasks = self._simulated.decompose(task)
            self._cache[task] = subtasks
            return list(subtasks)
        self.memory.add_experience(task, subtasks)
        self._cache[task] = subtasks
        return list(subtasks)

    def _parse(self, raw: str) -> list[Subtask]:
        """解析 LLM 输出为 Subtask 列表；非法项丢弃，全非法则抛异常。"""
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM 输出中不含 JSON 数组")
        items = json.loads(raw[start:end + 1])
        subtasks: list[Subtask] = []
        for it in items:
            task_type = str(it.get("task_type", "")).strip()
            if task_type not in ALLOWED_TYPES:
                continue
            quality = min(1.0, max(0.0, float(it.get("required_min_quality", 0.6))))
            cost = min(s.cost_per_call
                       for s in self.registry.list_by_type(task_type))
            desc = str(it.get("desc", "")).strip() or f"{task_type} 子任务"
            subtasks.append(Subtask(
                order=len(subtasks) + 1,
                task_type=task_type,
                required_min_quality=quality,
                estimated_cost=cost,
                desc=desc,
            ))
        if not subtasks:
            raise ValueError("LLM 输出中没有合法子任务")
        return subtasks
