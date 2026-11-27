"""Experience graph memory (long-term memory layer).

图结构长期记忆：跨实例、跨运行复用任务分解经验，减少 LLM 调用（省 token）。

结构：
    task 节点 ──decomposed_to(order=i)──> step 节点（含子任务快照）
    step 节点 ──served_by(calls/质量/开销累计)──> service 节点

能力：
- lookup(task)：先精确匹配，再用字符 bigram Jaccard 词面相似度找最相似的
  历史任务；命中即返回其分解链（省一次 LLM 调用），并附 service_hints
  （各步骤历史上表现最好的服务，供后续选择器/多 agent 参考）；
- add_experience / record_run_outputs：写入新经验；运行结束后回填
  「哪类步骤实际由哪个服务完成、质量/开销如何」，图越用越准；
- JSON 持久化（data/mock/experience_graph.json），进程重启后经验仍在；
- 内置 threading.Lock，多 agent 并发读写安全（为多智能体黑板做准备）。

零第三方依赖（re/json/threading 标准库），不接入向量库；
词面相似度阈值可调，接口预留升级空间。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .task_planner import Subtask

DEFAULT_MEMORY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "mock" / "experience_graph.json"
)

#: 词面相似度命中阈值（0~1；bigram Jaccard）
DEFAULT_SIMILARITY_THRESHOLD = 0.55

# 去掉标点/空白后做 bigram，避免标点干扰相似度
_PUNCT_RE = re.compile(r"[\s，。、；：？！\"'“”‘’（()）【】《》,.;:?!\-—_/\\]+")


def _bigrams(text: str) -> set[str]:
    t = _PUNCT_RE.sub("", text.lower())
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similarity(a: str, b: str) -> float:
    """字符 bigram Jaccard 相似度（0~1）。"""
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _normalize(task: str) -> str:
    return _PUNCT_RE.sub("", task.lower())


@dataclass(frozen=True)
class MemoryHit:
    """一次记忆检索命中。"""

    source_task: str                       # 经验来源任务原文
    similarity: float                      # 与查询任务的词面相似度（精确命中为 1.0）
    exact: bool                            # 是否精确命中
    subtasks: list[Subtask]                # 复用的分解链（order 已归一化为 1..n）
    service_hints: dict[str, dict] = field(default_factory=dict)
    # service_hints: {task_type: {service, avg_quality, avg_cost, calls}}——
    # 该类步骤历史上表现最好的服务（按平均质量），供选择器/多 agent 参考


class ExperienceGraph:
    """图结构经验记忆（任务 -> 分解链 -> 服务统计），JSON 持久化。"""

    def __init__(
        self,
        path: str | Path = DEFAULT_MEMORY_PATH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._path = Path(path)
        self._threshold = similarity_threshold
        self._lock = threading.Lock()
        # 节点: {id: {id, kind, label, ...}}；边: [{src, dst, rel, ...}]
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []
        self.total_hits = 0            # 累计命中次数（含精确/相似）
        self._load()

    # ---------- 写入 ----------

    def add_experience(self, task: str, subtasks: list[Subtask]) -> None:
        """记录一次任务分解经验（task 节点 + 有序分解链边 + step 节点）。"""
        with self._lock:
            tnode = self._ensure_task_node(task)
            for st in subtasks:
                snode = self._ensure_node("step", st.task_type)
                self._edges.append({
                    "src": tnode["id"], "dst": snode["id"], "rel": "decomposed_to",
                    "order": st.order,
                    "subtask": {
                        "order": st.order, "task_type": st.task_type,
                        "required_min_quality": st.required_min_quality,
                        "estimated_cost": st.estimated_cost, "desc": st.desc,
                    },
                })
            self._save()

    def record_run_outputs(self, task: str, steps: list) -> None:
        """运行结束后回填服务使用统计（step -served_by-> service 边累计）。

        steps 为 RunReport.steps（StepLog 序列，鸭子类型）：只统计
        executed 且 success 的步骤（失败调用计入质量均值会失真）。
        """
        with self._lock:
            for s in steps:
                if not (getattr(s, "executed", False) and getattr(s, "success", False)):
                    continue
                snode = self._ensure_node("step", s.task_type)
                vnode = self._ensure_node("service", s.service_name)
                edge = next((
                    e for e in self._edges
                    if e["src"] == snode["id"] and e["dst"] == vnode["id"]
                    and e["rel"] == "served_by"
                ), None)
                if edge is None:
                    edge = {"src": snode["id"], "dst": vnode["id"], "rel": "served_by",
                            "calls": 0, "total_quality": 0.0, "total_cost": 0.0}
                    self._edges.append(edge)
                edge["calls"] += 1
                edge["total_quality"] += float(s.quality)
                edge["total_cost"] += float(s.cost)
            # 关联到任务节点（便于追溯这次经验来自哪次运行）
            self._ensure_task_node(task)
            self._save()

    # ---------- 检索 ----------

    def lookup(self, task: str) -> MemoryHit | None:
        """查记忆：精确命中 -> 相似命中（>= threshold）；未命中返回 None。"""
        with self._lock:
            exact = self._find_exact(task)
            if exact is not None:
                self.total_hits += 1
                return self._build_hit(exact, 1.0, True)
            similar, sim = self._find_similar(task)
            if similar is not None:
                self.total_hits += 1
                return self._build_hit(similar, sim, False)
            return None

    def _find_exact(self, task: str) -> dict | None:
        nid = f"task::{_normalize(task)}"
        return self._nodes.get(nid)

    def _find_similar(self, task: str) -> tuple[dict | None, float]:
        best, best_sim = None, 0.0
        for node in self._nodes.values():
            if node["kind"] != "task":
                continue
            s = similarity(task, node["label"])
            if s > best_sim:
                best, best_sim = node, s
        if best is not None and best_sim >= self._threshold:
            return best, best_sim
        return None, 0.0

    def _chain_of(self, task_node_id: str) -> list[Subtask]:
        """从分解链边还原子任务序列（按 order 排序，order 归一化 1..n）。"""
        items = sorted(
            (e for e in self._edges
             if e["src"] == task_node_id and e["rel"] == "decomposed_to"),
            key=lambda e: e["order"],
        )
        return [
            Subtask(
                order=i + 1,
                task_type=it["subtask"]["task_type"],
                required_min_quality=float(it["subtask"]["required_min_quality"]),
                estimated_cost=float(it["subtask"]["estimated_cost"]),
                desc=str(it["subtask"]["desc"]),
            )
            for i, it in enumerate(items)
        ]

    def _hints_for(self, task_node_id: str) -> dict[str, dict]:
        """为命中任务的每个步骤类别汇总历史上最好的服务。"""
        types = {
            e["subtask"]["task_type"]
            for e in self._edges
            if e["src"] == task_node_id and e["rel"] == "decomposed_to"
        }
        hints: dict[str, dict] = {}
        for t in types:
            best = None
            for e in self._edges:
                if e["rel"] != "served_by" or not e["src"].endswith(f"step::{t}"):
                    continue
                if e["calls"] == 0:
                    continue
                avg_q = e["total_quality"] / e["calls"]
                if best is None or avg_q > best[0]:
                    best = (avg_q, e["dst"].removeprefix("service::"),
                            e["total_cost"] / e["calls"], e["calls"])
            if best is not None:
                hints[t] = {
                    "service": best[1], "avg_quality": round(best[0], 4),
                    "avg_cost": round(best[2], 4), "calls": best[3],
                }
        return hints

    def _build_hit(self, task_node: dict, sim: float, exact: bool) -> MemoryHit:
        nid = task_node["id"]
        return MemoryHit(
            source_task=task_node["label"], similarity=sim, exact=exact,
            subtasks=self._chain_of(nid),
            service_hints=self._hints_for(nid),
        )

    # ---------- 节点与持久化 ----------

    def _ensure_node(self, kind: str, label: str) -> dict:
        nid = f"{kind}::{label}"
        node = self._nodes.get(nid)
        if node is None:
            node = {"id": nid, "kind": kind, "label": label}
            self._nodes[nid] = node
        return node

    def _ensure_task_node(self, task: str) -> dict:
        nid = f"task::{_normalize(task)}"
        node = self._nodes.get(nid)
        if node is None:
            node = {"id": nid, "kind": "task", "label": task}
            self._nodes[nid] = node
        return node

    def stats(self) -> dict:
        """图规模概览（节点按类型计数 + 边按关系计数 + 命中累计）。"""
        with self._lock:
            n_by_kind: dict[str, int] = {}
            for node in self._nodes.values():
                n_by_kind[node["kind"]] = n_by_kind.get(node["kind"], 0) + 1
            e_by_rel: dict[str, int] = {}
            for e in self._edges:
                e_by_rel[e["rel"]] = e_by_rel.get(e["rel"], 0) + 1
            return {"nodes": n_by_kind, "edges": e_by_rel,
                    "total_hits": self.total_hits, "file": str(self._path)}

    def service_stats(self) -> list[dict]:
        """全部 served_by 统计（step -> service 的历史表现）。"""
        with self._lock:
            rows = []
            for e in self._edges:
                if e["rel"] != "served_by" or e["calls"] == 0:
                    continue
                rows.append({
                    "task_type": e["src"].removeprefix("step::"),
                    "service": e["dst"].removeprefix("service::"),
                    "calls": e["calls"],
                    "avg_quality": round(e["total_quality"] / e["calls"], 4),
                    "avg_cost": round(e["total_cost"] / e["calls"], 4),
                })
            rows.sort(key=lambda r: (-r["calls"], r["task_type"]))
            return rows

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
            "total_hits": self.total_hits,
        }
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._nodes = {n["id"]: n for n in data.get("nodes", [])}
            self._edges = list(data.get("edges", []))
            self.total_hits = int(data.get("total_hits", 0))
        except (OSError, ValueError, KeyError, TypeError):
            # 记忆文件损坏时静默从零开始，不影响主流程
            self._nodes, self._edges, self.total_hits = {}, [], 0
