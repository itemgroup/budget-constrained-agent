"""Retrieval over historical task runs (RAG for future tasks).

检索对象是**历史任务运行记录**（src/run_log.py 落盘的 data/run_log.jsonl），
而不是任何静态文档：每次运行结束都会沉淀一条记录，下一次遇到相似任务时，
先检索"以前这类任务怎么拆、花了多少钱、选了哪些服务"，作为分解与选型的参考。

两种检索后端，接口一致：
- tfidf（默认，纯标准库）：字符 bigram + ASCII 词 的 TF-IDF 向量 + 余弦相似度，
  无需下载模型、毫秒级、完全离线；
- chroma（可选向量库）：持久化到 kb_chroma/，需显式设置 RAG_BACKEND=chroma。

记录是持续增长的，检索前会检测文件变化并自动重建索引。
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .run_log import RUN_LOG_PATH

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = ROOT / "kb_chroma"
COLLECTION_NAME = "task_runs"


@dataclass(frozen=True)
class RagHit:
    """一条历史任务检索结果。"""

    index: int
    text: str
    score: float
    meta: dict = field(default_factory=dict)


#: 中英词表映射：任务描述多为中文，记录中服务/步骤名含英文，双向补词提升召回
_ZH_TO_EN: dict[str, tuple[str, ...]] = {
    "预算": ("budget",),
    "成本": ("cost",),
    "开销": ("cost", "spending"),
    "价格": ("price", "pricing"),
    "服务": ("service",),
    "调用": ("call",),
    "选择": ("selection", "select"),
    "分解": ("decomposition",),
    "任务": ("task",),
    "失败": ("failure", "fail"),
    "重试": ("retry",),
    "质量": ("quality",),
    "报告": ("report",),
    "检索": ("search",),
    "文档": ("document",),
    "分析": ("analysis",),
    "评估": ("evaluation",),
    "智能体": ("agent",),
    "约束": ("constrained",),
    "行程": ("travel", "itinerary"),
    "酒店": ("hotel",),
    "航班": ("flight",),
    "餐饮": ("food",),
    "活动": ("activity",),
}


def _expand_zh(text: str) -> list[str]:
    return [w for zh, words in _ZH_TO_EN.items() if zh in text for w in words]


def _tokens(text: str) -> list[str]:
    """中英混合分词：去标点后取字符 bigram，补充 ASCII 单词与中文关键词扩展。"""
    compact = re.sub(r"[\s\W_]+", "", text.lower(), flags=re.UNICODE)
    bigrams = [compact[i:i + 2] for i in range(len(compact) - 1)]
    words = re.findall(r"[a-z0-9]+", text.lower())
    return bigrams + words + _expand_zh(text)


class _TfidfBackend:
    """纯标准库 TF-IDF 向量检索（离线、确定性）。"""

    name = "tfidf"

    def __init__(self, docs: list[str]) -> None:
        self._docs = docs
        doc_tokens = [Counter(_tokens(d)) for d in docs]
        df: Counter = Counter()
        for counter in doc_tokens:
            df.update(set(counter))
        n = max(1, len(docs))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vectorize(c) for c in doc_tokens]

    def _vectorize(self, counter: Counter) -> dict[str, float]:
        vec = {t: tf * self._idf.get(t, 1.0) for t, tf in counter.items()
               if t in self._idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int) -> list[RagHit]:
        qvec = self._vectorize(Counter(_tokens(query)))
        scored: list[tuple[float, int]] = []
        for i, dvec in enumerate(self._vectors):
            small, big = (dvec, qvec) if len(dvec) < len(qvec) else (qvec, dvec)
            score = sum(v * big.get(t, 0.0) for t, v in small.items())
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [RagHit(i, self._docs[i], round(s, 4)) for s, i in scored[:k]]


class _ChromaBackend:
    """ChromaDB 向量检索（可选；记录更新时整库重建）。"""

    name = "chroma"

    def __init__(self, docs: list[str], persist_dir: Path, collection: str) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection)
        except Exception:  # noqa: BLE001 - 不存在时忽略
            pass
        self._col = client.create_collection(collection)
        if docs:
            self._col.add(documents=docs,
                          ids=[f"run-{i}" for i in range(len(docs))])
        self._docs = docs

    def search(self, query: str, k: int) -> list[RagHit]:
        if not self._docs:
            return []
        res = self._col.query(query_texts=[query], n_results=min(k, len(self._docs)))
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        hits: list[RagHit] = []
        for pos, doc in enumerate(docs):
            try:
                idx = int(str(ids[pos]).split("-")[-1])
            except (ValueError, IndexError):
                idx = pos
            dist = float(dists[pos]) if pos < len(dists) else 1.0
            hits.append(RagHit(idx, doc, round(max(0.0, 1.0 - dist), 4)))
        return hits


def _render_record(rec: dict) -> str:
    """把一条运行记录渲染为可检索文本。"""
    steps = rec.get("steps") or []
    lines = [
        f"任务：{rec.get('task', '')}",
        f"预算：${float(rec.get('budget', 0) or 0):.2f} "
        f"实际花费：${float(rec.get('spent', 0) or 0):.2f} "
        f"状态：{rec.get('state', '')}",
    ]
    for i, s in enumerate(steps, start=1):
        lines.append(
            f"{i}. [{s.get('type', '')}] {s.get('desc', '')} → {s.get('service', '')}"
            f"（${float(s.get('cost', 0) or 0):.2f}，质量 "
            f"{float(s.get('quality', 0) or 0):.2f}）"
        )
    if rec.get("skipped"):
        lines.append("跳过：" + "、".join(str(x) for x in rec["skipped"]))
    judge = rec.get("judge")
    if judge:
        lines.append(f"质量评分：{judge.get('overall')}")
    audit = rec.get("audit")
    if audit:
        lines.append("审计：" + ("通过" if audit.get("passed") else "驳回"))
    return "\n".join(lines)


#: 任务文本以这些词开头视为演示 / 测试任务（防御历史遗留记录）
_MOCK_TASK_PREFIXES = ("mock", "demo", "示例", "演示", "测试")


def _is_mock_record(rec: dict) -> bool:
    """判断一条运行记录是否属于模拟 / 演示数据（是则不进入检索库）。

    规则（任一命中即视为 mock）：
    1. mode 字段不是 "llm"（历史记录缺少 mode 时也按 mock 处理）；
    2. 任务文本为空，或以 mock / demo / 示例 / 演示 / 测试 开头。
    """
    if str(rec.get("mode", "")).strip().lower() != "llm":
        return True
    task = str(rec.get("task", "")).strip().lower()
    if not task:
        return True
    return task.startswith(_MOCK_TASK_PREFIXES)


class RAGRetriever:
    """历史任务检索门面：统一 search / context / stats。

    索引前会过滤 mock 数据（见 _is_mock_record）：模拟运行、演示数据、
    LLM 回退以及历史遗留的无 mode 记录都不会被切片进检索库。
    调试时可传 include_mock=True 关闭过滤。
    """

    def __init__(
        self,
        path: str | Path = RUN_LOG_PATH,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        backend: str | None = None,
        k: int = 3,
        include_mock: bool = False,
    ) -> None:
        self.path = Path(path)
        self.persist_dir = Path(persist_dir)
        self.k = k
        self.include_mock = include_mock
        self._mtime: float | None = None
        self._records: list[dict] = []
        self._backend = None
        self._mock_skipped = 0
        self._invalid_skipped = 0
        self._mode = (backend or os.environ.get("RAG_BACKEND") or "auto").lower()
        self._reload()

    # ---------- 索引 ----------

    def _load_records(self) -> list[dict]:
        """加载并过滤记录：mock / 演示数据不计入检索切片。"""
        self._mock_skipped = 0
        self._invalid_skipped = 0
        if not self.path.exists():
            return []
        records: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                self._invalid_skipped += 1
                continue
            if not isinstance(rec, dict):
                self._invalid_skipped += 1
                continue
            if not self.include_mock and _is_mock_record(rec):
                self._mock_skipped += 1
                continue
            if not str(rec.get("task", "")).strip():
                self._invalid_skipped += 1
                continue
            records.append(rec)
        return records

    def _reload(self) -> None:
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime == self._mtime and self._backend is not None:
            return
        self._mtime = mtime
        self._records = self._load_records()
        docs = [_render_record(r) for r in self._records]
        if self._mode == "chroma":
            self._backend = _ChromaBackend(docs, self.persist_dir, COLLECTION_NAME)
        elif self._mode == "tfidf":
            self._backend = _TfidfBackend(docs)
        else:  # auto：只有本地库已建好时才用嵌入后端，避免启动期下载模型
            if self.persist_dir.exists():
                try:
                    self._backend = _ChromaBackend(docs, self.persist_dir,
                                                   COLLECTION_NAME)
                except Exception:  # noqa: BLE001
                    self._backend = _TfidfBackend(docs)
            else:
                self._backend = _TfidfBackend(docs)

    # ---------- 对外接口 ----------

    def search(self, query: str, k: int | None = None) -> list[RagHit]:
        """检索与 query 最相似的历史任务（分数降序）。"""
        q = (query or "").strip()
        if not q:
            return []
        self._reload()
        hits = self._backend.search(q, k or self.k)
        for hit in hits:
            rec = self._records[hit.index] if hit.index < len(self._records) else {}
            hit.meta.update({
                "task": rec.get("task", ""),
                "budget": rec.get("budget"),
                "spent": rec.get("spent"),
                "state": rec.get("state", ""),
            })
        return hits

    def context(self, query: str, k: int | None = None, max_chars: int = 1200) -> str:
        """把检索结果拼成提示词上下文（自动截断）。"""
        parts: list[str] = []
        used = 0
        for i, hit in enumerate(self.search(query, k), start=1):
            snippet = hit.text.strip()
            if used + len(snippet) > max_chars:
                snippet = snippet[: max(0, max_chars - used)]
            if not snippet:
                break
            parts.append(f"[{i}] {snippet}")
            used += len(snippet)
            if used >= max_chars:
                break
        return "\n".join(parts)

    def stats(self) -> dict:
        self._reload()  # 记录随时增长，统计前先同步
        return {
            "backend": self._backend.name if self._backend else None,
            "records": len(self._records),
            "mock_skipped": self._mock_skipped,
            "invalid_skipped": self._invalid_skipped,
            "include_mock": self.include_mock,
            "persist_dir": str(self.persist_dir),
        }
