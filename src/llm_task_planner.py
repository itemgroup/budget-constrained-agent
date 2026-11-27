"""LLM-driven task planner (LangChain + OpenAI-compatible endpoint).

真实 LLM 版任务分解器，接口与 TaskPlanner 完全一致：
    decompose(task: str) -> list[Subtask]
可注入 AgentOrchestrator(planner=...) 无缝替换模拟版（关键词匹配）。

设计要点：
- LLM 只负责"理解与分解"，输出 task_type / required_min_quality / desc；
- estimated_cost 不信任 LLM，按类别最低服务价从注册表确定性填充；
- 两级缓存省 token：实例内精确缓存（同任务字符串只调一次 LLM）+ 图结构
  长期记忆（src/memory.py，跨实例/跨运行的精确与相似命中均跳过 LLM 调用，
  并把每次分解经验持久化到 data/mock/experience_graph.json）；use_memory=False 时
  跳过图记忆，确保真实模型被实际调用；
- RAG（src/rag.py）：注入 retriever 后，分解前先检索知识库，把相关切片
  作为「参考资料」拼进提示词；检索结果是本地数据，不额外计费；
- 配置来源两级：显式传入 LLMConfig（前端运行时可配、构造即校验）优先，
  否则读环境变量 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL；
- 无密钥或 LLM 调用失败时回退到模拟版 TaskPlanner（fallback=True，默认开启）。

token 计数：llm_calls（实际发起的 LLM 调用次数）、memory_hits（记忆命中
省下的调用次数）——demo/前端可据此展示"记忆省了多少 token"。
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from .llm_config import LLMConfig
from .memory import ExperienceGraph
from .rag import RAGRetriever
from .service_registry import ServiceRegistry
from .task_planner import Subtask, TaskPlanner

#: 允许的任务类型（与 Service.service_type 对应）
ALLOWED_TYPES = ("Search", "Document Parsing", "LLM Reasoning", "Report Generation")

#: 分解提示词：要求严格输出 JSON 数组；{context} 为空串时不注入参考资料
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

{context}任务：{task}"""

_CONTEXT_HEADER = "参考历史任务（相似任务的做法与花费，仅供借鉴）：\n"


class LLMTaskPlanner:
    """基于 LangChain + OpenAI 兼容端点的任务分解器（与 TaskPlanner 接口一致）。"""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        fallback: bool = True,
        memory: ExperienceGraph | None = None,
        config: LLMConfig | None = None,
        rag: RAGRetriever | None = None,
        use_memory: bool = True,
    ) -> None:
        load_dotenv()
        self.registry = registry if registry is not None else ServiceRegistry()
        self.fallback = fallback
        self._simulated = TaskPlanner()
        self._cache: dict[str, list[Subtask]] = {}
        self._chain = None
        self.rag = rag
        self.last_rag_hits = []
        # 是否启用图记忆复用；前端勾选「忽略历史记忆」时置 False，确保真实调用
        self.use_memory = use_memory
        # 图结构长期记忆：跨实例/跨运行复用分解经验，命中即省一次 LLM 调用
        self.memory = memory if memory is not None else ExperienceGraph()
        self.llm_calls = 0      # 实际发起的 LLM 调用次数
        self.llm_success = 0    # 成功返回并解析的分解次数
        self.memory_hits = 0    # 记忆命中（省掉的 LLM 调用）次数
        self.config: LLMConfig | None = None

        cfg = config or self._config_from_env(model=model)
        if cfg is None:
            msg = "未找到 DEEPSEEK_API_KEY（请在 .env 中配置，或在前端 LLM 配置中填写）"
            if not fallback:
                raise RuntimeError(msg)
            print(f"[LLMTaskPlanner] {msg}，使用模拟分解器")
            return

        self.config = cfg
        # 延迟导入：未使用真实 LLM 时不要求 langchain 可用
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature if cfg.temperature else temperature,
        )
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self._chain = prompt | llm | StrOutputParser()

    @staticmethod
    def _config_from_env(model: str) -> LLMConfig | None:
        """从环境变量构造配置；无密钥返回 None（走模拟分解器）。"""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model_name = os.environ.get("DEEPSEEK_MODEL", model)
        # 配置不合规时直接抛出，避免带着错误端点静默失败
        return LLMConfig(base_url=base_url, api_key=api_key, model=model_name)

    # ---------- 分解 ----------

    def decompose(self, task: str) -> list[Subtask]:
        """把自然语言任务分解为有序子任务列表（实例缓存 + 图记忆两级复用）。"""
        if task in self._cache:
            return list(self._cache[task])

        # 1) 图记忆：精确/相似命中均直接复用历史分解，跳过 LLM 调用
        #    use_memory=False 时跳过该步骤，确保真实 LLM 被实际调用
        hit = self.memory.lookup(task) if self.use_memory else None
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

        # 3) RAG 检索（本地知识库，零外部调用）
        context = ""
        if self.rag is not None:
            self.last_rag_hits = self.rag.search(task)
            context = self._format_context(self.last_rag_hits)

        # 4) 真实 LLM 分解；失败回退模拟（回退结果不入记忆，避免污染经验库）
        self.llm_calls += 1
        try:
            raw = self._chain.invoke({"task": task, "context": context})
            subtasks = self._parse(raw)
        except Exception as e:
            if not self.fallback:
                raise
            print(f"[LLMTaskPlanner] LLM 分解失败（{e}），回退到模拟分解器")
            subtasks = self._simulated.decompose(task)
            self._cache[task] = subtasks
            return list(subtasks)
        self.memory.add_experience(task, subtasks)
        self.llm_success += 1
        self._cache[task] = subtasks
        return list(subtasks)

    # ---------- 内部工具 ----------

    @staticmethod
    def _format_context(hits, max_chars: int = 1200) -> str:
        """把检索结果拼成提示词上下文（带表头，自动截断）。"""
        if not hits:
            return ""
        parts: list[str] = []
        used = 0
        for i, hit in enumerate(hits, start=1):
            snippet = hit.text.strip()
            if used + len(snippet) > max_chars:
                snippet = snippet[: max(0, max_chars - used)]
            if snippet:
                parts.append(f"[{i}] {snippet}")
                used += len(snippet)
            if used >= max_chars:
                break
        if not parts:
            return ""
        return _CONTEXT_HEADER + "\n".join(parts) + "\n\n"

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
