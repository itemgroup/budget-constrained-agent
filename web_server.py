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

运行方式（项目根目录下）：
    python web_server.py
然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.agent_core import AgentOrchestrator
from src.evaluation.evaluator import AGENT_NAMES, Evaluator, build_agent, load_tasks
from src.scenarios.scenario import build_travel_agent
from src.service_registry import ServiceRegistry
from src.simulated_services import SimulatedServiceRunner, load_failure_rates

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "frontend" / "index.html"
RESULTS_DIR = ROOT / "experiments" / "results"
HOST, PORT = "127.0.0.1", 8000

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

    def _parse_run_params(self) -> tuple[str, float, int]:
        """解析并校验运行参数（/api/run 与 /api/compare 共用）。"""
        payload = self._read_json_body()
        task = str(payload.get("task", "")).strip()
        budget = float(payload.get("budget", 1.0))
        seed = int(payload.get("seed", 0))
        if not task:
            raise ValueError("任务描述不能为空")
        if budget <= 0:
            raise ValueError("预算必须为正数")
        return task, budget, seed

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/tasks":
            tasks, tiers = load_tasks()
            self._send_json({"tasks": tasks, "budget_tiers": tiers})
        elif self.path == "/api/services":
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
        elif self.path == "/api/experiments":
            datasets = {}
            for name in SUMMARY_CSVS:
                path = RESULTS_DIR / name
                if path.exists():
                    with open(path, encoding="utf-8-sig", newline="") as f:
                        datasets[name] = list(csv.DictReader(f))
            self._send_json({"available": bool(datasets), "datasets": datasets})
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/run":
            self._handle_run()
        elif self.path == "/api/compare":
            self._handle_compare()
        elif self.path == "/api/travel":
            self._handle_travel()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _handle_run(self) -> None:
        try:
            task, budget, seed = self._parse_run_params()
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"参数错误: {e}"}, status=400)
            return
        agent = AgentOrchestrator(runner=SimulatedServiceRunner(seed=seed))
        report = agent.run(task, budget)
        data = asdict(report)
        data["state"] = report.state.value  # 枚举转为可读字符串
        self._send_json(data)

    def _handle_compare(self) -> None:
        try:
            task, budget, seed = self._parse_run_params()
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
        """行程规划场景：实时报价单 + 预订执行（复用同一引擎，零引擎修改）。"""
        try:
            task, budget, seed = self._parse_run_params()
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"参数错误: {e}"}, status=400)
            return
        agent, registry = build_travel_agent(seed=seed)
        report = agent.run(task, budget)
        data = asdict(report)
        data["state"] = report.state.value  # 枚举转为可读字符串
        data["price_sheet"] = [
            {
                "name": s.name,
                "service_type": s.service_type,
                "cost_per_call": s.cost_per_call,
                "quality_score": s.quality_score,
            }
            for s in registry.list_all()
        ]
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
