"""Persist task run records for future retrieval (append-only JSONL).

每次任务运行结束后追加一条记录（任务、预算、实际花费、各步骤选用的服务与结果、
评分/审计结论）。这些记录就是 RAG 的知识来源：未来的相似任务可以检索到
"以前这类任务是怎么拆的、花了多少钱、用了哪个服务"，从而少走弯路、少花 token。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_LOG_PATH = ROOT / "data" / "run_log.jsonl"

_LOCK = threading.Lock()


def append_run(record: dict) -> dict:
    """追加一条运行记录（自动补时间戳），返回落盘后的记录。"""
    item = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


def record_from_report(
    task: str,
    budget: float,
    report,
    judge=None,
    audit=None,
    mode: str = "mock",
) -> dict:
    """把一次运行报告整理成可检索、可读的记录。

    mode 区分记录来源：只有 "llm"（真实 LLM 参与并成功）才会进入检索库，
    "mock"（纯模拟 / 演示 / LLM 回退）不入库，避免演示数据污染参考。
    """
    state = report.state.value if hasattr(report.state, "value") else str(report.state)
    steps = [
        {
            "type": s.task_type,
            "desc": s.subtask_desc,
            "service": s.service_name,
            "cost": round(float(s.cost or 0.0), 4),
            "quality": round(float(s.quality or 0.0), 4),
        }
        for s in report.steps
        if getattr(s, "executed", False)
    ]
    record = {
        "task": task,
        "budget": round(float(budget), 2),
        "spent": round(float(report.spent), 2),
        "state": state,
        "mode": mode if mode in ("llm", "mock") else "mock",
        "steps": steps,
        "skipped": list(report.skipped_subtasks),
    }
    if judge is not None:
        record["judge"] = {"overall": round(float(judge.overall), 4)}
    if audit is not None:
        record["audit"] = {"passed": bool(audit.passed), "reason": audit.reason}
    return record
