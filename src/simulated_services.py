"""Simulated tool calls, random failure simulation.

模拟服务执行层：接收上层选中的服务与子任务参数，模拟一次真实服务调用。

- 按 configs/service_config.yaml 中各服务的 failure_probability 做随机失败；
- 成功返回模拟输出与本次实际开销；失败返回失败标记（失败不扣费，cost=0.0）；
- 与 BudgetLedger 解耦：本模块不记账，扣费记账由上层 agent 负责。

仅做模拟，不调用真实 LLM、不接入 RAG。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .service_registry import Service
from .task_planner import Subtask

# 内置默认失败率表，数据与 docs/proposal.md 3.2 节一致。
# 仅当 configs/service_config.yaml 不可用（或未安装 pyyaml）时作为回退。
DEFAULT_FAILURE_RATES: dict[str, float] = {
    "basic_search": 0.05,
    "premium_search": 0.02,
    "simple_parser": 0.05,
    "advanced_parser": 0.03,
    "small_llm": 0.04,
    "large_llm": 0.01,
    "template_report": 0.03,
    "custom_report": 0.02,
}

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "service_config.yaml"
)


@dataclass(frozen=True)
class ServiceResult:
    """一次模拟服务调用的结构化结果。"""

    success: bool   # 是否成功
    output: str     # 模拟输出；失败时为空字符串
    cost: float     # 本次实际开销（美元）；失败时为 0.0（不扣费）
    quality: float  # 本次产出质量分；失败时为 0.0
    error_msg: str  # 失败原因；成功时为空字符串


def load_failure_rates(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> dict[str, float]:
    """从 YAML 配置加载各服务失败率；pyyaml 不可用或文件缺失时回退到内置默认表。"""
    try:
        import yaml
    except ImportError:
        return dict(DEFAULT_FAILURE_RATES)
    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {
            item["name"]: float(item["failure_probability"])
            for item in data["services"]
        }
    except (OSError, KeyError, TypeError):
        return dict(DEFAULT_FAILURE_RATES)


class SimulatedServiceRunner:
    """模拟服务执行器：对上层选中的服务执行一次模拟调用。"""

    def __init__(
        self,
        failure_rates: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        # failure_rates 由外部注入（便于实验脚本做失败注入测试）；
        # 缺省时自动从 configs/service_config.yaml 加载
        self._failure_rates = (
            failure_rates if failure_rates is not None else load_failure_rates()
        )
        # 独立随机源：seed 可复现实验，且不影响全局 random 状态
        self._rng = random.Random(seed)

    def failure_probability(self, service_name: str) -> float:
        """查询某服务的配置失败率；未知服务抛出 KeyError。"""
        try:
            return self._failure_rates[service_name]
        except KeyError:
            raise KeyError(f"未知服务失败率: {service_name}") from None

    def run(self, service: Service, subtask: Subtask | None = None) -> ServiceResult:
        """执行一次模拟调用。

        参数:
            service: 上层选择器选中的服务（提供价格、质量分等元数据）
            subtask: 当前子任务参数（可选，仅用于生成模拟输出文本）
        返回:
            ServiceResult（成功/失败标记、模拟输出、实际开销、质量分、失败原因）
        """
        p = self.failure_probability(service.name)
        if self._rng.random() < p:
            return ServiceResult(
                success=False,
                output="",
                cost=0.0,
                quality=0.0,
                error_msg=f"{service.name} 模拟随机失败（配置失败率 {p:.2f}）",
            )
        desc = subtask.desc if subtask is not None else "（未指定子任务）"
        output = (
            f"[模拟输出] {service.name} 已完成 "
            f"[{service.service_type}] 子任务：{desc}"
        )
        return ServiceResult(
            success=True,
            output=output,
            cost=service.cost_per_call,
            quality=service.quality_score,
            error_msg="",
        )
