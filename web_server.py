"""Local web UI server (stdlib only, no third-party deps).

本地网页界面：仿 Apple 官网风格的分区式演示站（纯原生 HTML/CSS/JS，零外部资源）。
前端为 hash 路由单页应用，含 首页 / 运行 / 对比 / 服务目录 / 实验数据 / 关于 六个视图。

- GET  /                -> 前端页面 frontend/index.html
- GET  /api/tasks       -> 预置测试任务列表（来自 evaluation/test_tasks.json）
- GET  /api/services    -> 服务注册表 + 各服务配置失败率
- GET  /api/experiments -> experiments/results 下的实验摘要 CSV（无文件时 available=false）
- POST /api/run         -> {task, budget, seed} 运行完整版智能体，返回运行报告 JSON
- POST /api/compare     -> {task, budget, seed} 同任务对比 4 种智能体
                           （standard / min_cost / hard_stop / proposed，相同 seed 保证公平）
- POST /api/travel      -> {task, budget, seed} 行程规划场景（扩展包）：先生成本次
                           实时报价单（基准价 ±15% 波动），再由同一引擎择优预订，返回
                           运行报告 + 报价单
- GET  /api/llm/status  -> 当前 LLM / RAG 状态（配置只返回掩码，绝不回传完整密钥）
- POST /api/llm/config  -> {base_url, api_key, model} 校验地址合规性后保存在进程内存
- POST /api/llm/test    -> {base_url, api_key, model} 地址合规性 + 真实连接校验
- POST /api/llm/clear   -> 清除运行时 LLM 配置
- POST /api/run 额外字段 -> use_llm（真实 LLM 分解+评分）、use_rag（检索知识库）

运行方式（项目根目录下）：
    python web_server.py
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import csv
import json
import os
import threading
from collections import Counter
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.agent_core import AgentOrchestrator
from src.evaluation.evaluator import AGENT_NAMES, Evaluator, build_agent, load_tasks
from src.evaluation.llm_judge import RealLLMJudge
from src.llm_config import LLMConfig, LLMConfigError, test_connection
from src.llm_task_planner import LLMTaskPlanner
from src.multi_agent.orchestrator import (
    build_enterprise_multi_agent,
    build_travel_multi_agent,
)
from src.rag import RAGRetriever
from src.run_log import append_run, record_from_report
from src.scenarios.scenario import (
    build_price_registry,
    build_travel_agent,
    load_catalog,
)
from src.service_registry import ServiceRegistry
from src.simulated_services import SimulatedServiceRunner, load_failure_rates

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "frontend" / "index.html"
RESULTS_DIR = ROOT / "experiments" / "results"
HOST = os.environ.get("WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("WEB_PORT", "8000"))

# 实验数据视图读取的摘要 CSV（与 experiments/*.py 的输出文件名一致）
SUMMARY_CSVS = (
    "baselines_summary.csv",
    "ablation_summary.csv",
    "budget_sweep_summary.csv",
    "failure_test_summary.csv",
)

AGENT_LABELS = {
    "standard": "Standard（质量优先，内部无预算约束）",
    "min_cost": "Minimum-Cost（只选最便宜的可支付服务）",
    "hard_stop": "Budget Hard-Stop（无降级，付不起即停）",
    "proposed": "Proposed（完整版：预算感知 + 动态重规划）",
}

# ---- 运行时 LLM / RAG 状态（只存在进程内存，不落盘、不写 .env） ----
_LLM_LOCK = threading.Lock()
_LLM_STATE: dict = {"config": None, "verified": False}
_RAG_STATE: dict = {"instance": None, "error": None}


def _get_rag() -> RAGRetriever | None:
    """惰性构建 RAG 检索器；失败时缓存错误，避免每次请求重复尝试。"""
    if _RAG_STATE["instance"] is None and _RAG_STATE["error"] is None:
        try:
            _RAG_STATE["instance"] = RAGRetriever()
        except Exception as e:  # noqa: BLE001 - 任何构建失败都降级为"不可用"
            _RAG_STATE["error"] = str(e)
    return _RAG_STATE["instance"]


def _env_config() -> LLMConfig | None:
    """从 .env / 环境变量读取配置；缺失或不合法时返回 None。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        return LLMConfig(
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=key,
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    except LLMConfigError:
        return None


def _resolve_config(payload: dict) -> LLMConfig:
    """构造配置：API Key 留空时沿用已保存的配置或 .env 中的密钥。"""
    data = dict(payload)
    if not str(data.get("api_key", "")).strip():
        prev = _LLM_STATE["config"]
        key = prev.api_key if prev is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        if key:
            data["api_key"] = key
    return LLMConfig.from_payload(data)


def _llm_status() -> dict:
    """对外状态：配置只返回掩码，RAG 返回后端与切片数。"""
    runtime = _LLM_STATE["config"]
    env_cfg = _env_config() if runtime is None else None
    cfg = runtime or env_cfg
    rag = _get_rag()
    if rag is not None:
        rag_info = rag.stats()
    else:
        rag_info = {"backend": None, "chunks": 0, "error": _RAG_STATE["error"]}
    return {
        "configured": cfg is not None,
        "verified": bool(_LLM_STATE["verified"]) if runtime is not None else False,
        "source": "runtime" if runtime is not None else ("env" if env_cfg else None),
        "config": cfg.masked() if cfg else None,
        "rag": rag_info,
    }


class Handler(BaseHTTPRequestHandler):
    """最小 API + 静态页处理器。"""

    # 静默默认访问日志，保持控制台干净
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _parse_run_params(self) -> tuple[str, float, int, dict]:
        """解析并校验运行参数（/api/run 与 /api/compare 共用）。"""
        payload = self._read_json_body()
        task = str(payload.get("task", "")).strip()
        budget = float(payload.get("budget", 1.0))
        seed = int(payload.get("seed", 0))
        if not task:
            raise ValueError("任务描述不能为空")
        if budget <= 0:
            raise ValueError("预算必须为正数")
        return task, budget, seed, payload

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/tasks":
            tasks, tiers = load_tasks()
            self._send_json({"tasks": tasks, "budget_tiers": tiers})
        elif path == "/api/services":
            rates = load_failure_rates()
            services = [
                {
                    "name": s.name,
                    "service_type": s.service_type,
                    "cost_per_call": s.cost_per_call,
                    "quality_score": s.quality_score,
                    "description": s.description,
                    "failure_probability": rates.get(s.name),
                }
                for s in ServiceRegistry().list_all()
            ]
            self._send_json({"services": services})
        elif path == "/api/experiments":
            datasets = {}
            for name in SUMMARY_CSVS:
                summary_path = RESULTS_DIR / name
                if summary_path.exists():
                    with open(summary_path, encoding="utf-8-sig", newline="") as f:
                        datasets[name] = list(csv.DictReader(f))
            self._send_json({"available": bool(datasets), "datasets": datasets})
        elif path == "/api/llm/status":
            self._send_json(_llm_status())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            self._handle_run()
        elif path == "/api/compare":
            self._handle_compare()
        elif path == "/api/travel":
            self._handle_travel()
        elif path == "/api/llm/config":
            self._handle_llm_config()
        elif path == "/api/llm/test":
            self._handle_llm_test()
        elif path == "/api/llm/clear":
            self._handle_llm_clear()
        else:
            self._send_json({"error": "not found"}, status=404)

    # ---------- LLM 运行时配置 ----------

    def _handle_llm_config(self) -> None:
        """保存配置：先做地址合规性校验，通过后仅存内存。"""
        try:
            cfg = _resolve_config(self._read_json_body())
        except (LLMConfigError, ValueError, TypeError) as e:
            self._send_json({"error": f"配置不合规: {e}"}, status=400)
            return
        with _LLM_LOCK:
            _LLM_STATE["config"] = cfg
            _LLM_STATE["verified"] = False
        self._send_json({"ok": True, "status": _llm_status()})

    def _handle_llm_test(self) -> None:
        """连接校验：地址合规性 + 一次最小真实请求。"""
        try:
            cfg = _resolve_config(self._read_json_body())
        except (LLMConfigError, ValueError, TypeError) as e:
            self._send_json({"error": f"配置不合规: {e}"}, status=400)
            return
        result = test_connection(cfg)
        if result["ok"]:
            with _LLM_LOCK:
                if _LLM_STATE["config"] == cfg:
                    _LLM_STATE["verified"] = True
        self._send_json({"ok": result["ok"], "result": result,
                         "config": cfg.masked(), "status": _llm_status()})

    def _handle_llm_clear(self) -> None:
        """清除运行时配置（回到模拟模式）。"""
        with _LLM_LOCK:
            _LLM_STATE["config"] = None
            _LLM_STATE["verified"] = False
        self._send_json({"ok": True, "status": _llm_status()})

    def _handle_run(self) -> None:
        try:
            task, budget, seed, payload = self._parse_run_params()
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"参数错误: {e}"}, status=400)
            return

        use_llm = bool(payload.get("use_llm"))
        use_rag = bool(payload.get("use_rag"))
        ignore_memory = bool(payload.get("ignore_memory"))
        planner = None
        judge = None
        cfg = None
        rag = None
        judge_score = None
        llm_meta = {
            "used": False, "decomposition": "simulated",
            "llm_calls": 0, "memory_hits": 0, "memory_ignored": ignore_memory,
            "rag": None, "judge": None,
        }

        if use_llm:
            cfg = _LLM_STATE["config"] or _env_config()
            if cfg is None:
                self._send_json(
                    {"error": "尚未配置 LLM：请在「LLM 与 RAG」卡片填写并保存，"
                              "或在项目根目录 .env 中设置 DEEPSEEK_API_KEY"},
                    status=400,
                )
                return
            if use_rag:
                rag = _get_rag()
                if rag is None:
                    self._send_json(
                        {"error": f"RAG 不可用: {_RAG_STATE['error']}"}, status=400
                    )
                    return
            try:
                planner = LLMTaskPlanner(
                    config=cfg, rag=rag, use_memory=not ignore_memory
                )
                judge = RealLLMJudge(config=cfg)
            except Exception as e:  # noqa: BLE001 - 依赖缺失或配置异常
                self._send_json({"error": f"LLM 初始化失败: {e}"}, status=400)
                return

        multi = bool(payload.get("multi", True))  # 默认多 agent 编排
        if multi:
            agent = build_enterprise_multi_agent(
                seed=seed, planner=planner, judge=judge
            )
            report = agent.run(task, budget)
            data = asdict(report)
            data["state"] = report.state.value
            audit = report.audit_report
            data["audit"] = {
                "passed": audit.passed,
                "reconciliation_ok": audit.reconciliation_ok,
                "format_errors": list(audit.format_errors),
                "reason": audit.reason,
                "score": asdict(audit.score) if audit.score else None,
            } if audit else None
            data["multi"] = True
        else:
            agent = AgentOrchestrator(
                planner=planner, runner=SimulatedServiceRunner(seed=seed)
            )
            report = agent.run(task, budget)
            data = asdict(report)
            data["state"] = report.state.value  # 枚举转为可读字符串
            data["audit"] = None
            data["multi"] = False

        if planner is not None and cfg is not None:
            hits = getattr(planner, "last_rag_hits", []) or []
            if planner.llm_success:
                decomposition = "llm"
            elif planner.llm_calls:
                decomposition = "fallback"
            else:
                decomposition = "simulated"
            llm_meta.update({
                "used": bool(planner.llm_success),
                "decomposition": decomposition,
                "llm_calls": planner.llm_calls,
                "memory_hits": planner.memory_hits,
                "memory_ignored": ignore_memory,
                "config": cfg.masked(),
                "rag": ({
                    "backend": rag.stats()["backend"],
                    "chunks": rag.stats()["chunks"],
                    "hits": [
                        {"score": h.score, "text": h.text[:120]} for h in hits
                    ],
                } if rag is not None else None),
            })
            planned = len(report.outputs) + len(report.skipped_subtasks)
            score = judge.score(task, report.outputs, planned)
            judge_score = score
            llm_meta["judge"] = {
                "completeness": score.completeness,
                "correctness": score.correctness,
                "overall": score.overall,
                "rationale": score.rationale,
                "fallback": score.rationale.startswith("[LLM 评分失败回退]"),
            }
        data["llm"] = llm_meta
        # 沉淀本次运行记录；只有真实 LLM 成功参与运行才入检索库（模拟/Mock 不计入）
        real_llm = bool(
            use_llm and planner is not None
            and getattr(planner, "llm_success", False)
        )
        try:
            append_run(record_from_report(
                task, budget, report, judge_score,
                getattr(report, "audit_report", None),
                mode="llm" if real_llm else "mock",
            ))
        except Exception:  # noqa: BLE001
            pass
        self._send_json(data)

    def _handle_compare(self) -> None:
        try:
            task, budget, seed, _payload = self._parse_run_params()
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"参数错误: {e}"}, status=400)
            return
        ev = Evaluator()
        task_entry = {"id": "web-custom", "task": task}
        results = []
        for name in AGENT_NAMES:
            # 同一任务下各智能体使用相同 seed，失败模式一致、对比公平
            agent = build_agent(name, seed=seed)
            m = ev.run_single(agent, name, task_entry, "custom", budget)
            d = m.to_dict()
            d["label"] = AGENT_LABELS[name]
            d["service_usage"] = dict(Counter(
                s.service_name for s in agent.steps if s.executed
            ))
            results.append(d)
        self._send_json({"task": task, "budget": budget, "results": results})

    def _handle_travel(self) -> None:
        """行程规划场景：实时报价单 + 预订执行（可选多 agent 编排）。"""
        try:
            task, budget, seed, payload = self._parse_run_params()
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"参数错误: {e}"}, status=400)
            return
        multi = bool(payload.get("multi", True))  # 默认多 agent 编排
        if multi:
            agent = build_travel_multi_agent(seed=seed)
            report = agent.run(task, budget)
            registry, _rates = build_price_registry(load_catalog(), seed=seed)
            data = asdict(report)
            data["state"] = report.state.value
            audit = report.audit_report
            data["audit"] = {
                "passed": audit.passed,
                "reconciliation_ok": audit.reconciliation_ok,
                "format_errors": list(audit.format_errors),
                "reason": audit.reason,
                "score": asdict(audit.score) if audit.score else None,
            } if audit else None
            data["multi"] = True
        else:
            agent, registry = build_travel_agent(seed=seed)
            report = agent.run(task, budget)
            data = asdict(report)
            data["state"] = report.state.value  # 枚举转为可读字符串
            data["audit"] = None
            data["multi"] = False
        data["price_sheet"] = [
            {
                "name": s.name,
                "service_type": s.service_type,
                "cost_per_call": s.cost_per_call,
                "quality_score": s.quality_score,
            }
            for s in registry.list_all()
        ]
        try:
            append_run(record_from_report(
                task, budget, report, None,
                getattr(report, "audit_report", None),
                mode="mock",  # 行程场景为模拟核算，不计入检索库
            ))
        except Exception:  # noqa: BLE001
            pass
        self._send_json(data)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Web UI 已启动: http://{HOST}:{PORT}  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
