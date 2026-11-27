"""Consumption-scenario extension layer (beyond proposal scope).

场景扩展包（proposal 范围之外的可选演示）：证明同一套预算约束引擎
可以零修改地泛化到"真实花钱"的消费场景（行程规划、采购等）。

核心思路：
- 选项目录（YAML）直接装进现有 Service / ServiceRegistry——
  选项价格 = Service.cost_per_call，体验分 = Service.quality_score；
- BudgetAwareSelector 原封不动 => "预算内选体验最好的选项，紧张时降级"；
- Replanner 原封不动 => "满房/售罄 => 换次优选项（RETRY）/ 砍环节（SKIP）/ 放弃（ABORT）"；
- 实时报价：每次运行按 seed 对基准价施加 ±15% 波动，生成"本次查询到的报价单"。
  选择器、执行器、账本看到的是同一张报价单，因此"永不超支"不变量依然成立。

本模块只新增代码，不修改任何已有模块。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from ..agent_core import Subtask
from ..service_registry import Service, ServiceRegistry
from ..simulated_services import ServiceResult

# 价格波动幅度：本次报价 = 基准价 × U(0.85, 1.15)
PRICE_NOISE_LOW, PRICE_NOISE_HIGH = 0.85, 1.15

_DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "travel_catalog.yaml"
)


# ---------- 选项目录加载 ----------

@dataclass(frozen=True)
class PlanStepSpec:
    """行程模板中的一个环节规格。"""

    task_type: str            # 环节类别，如 Flight / Hotel / Food / Activity
    count: int                # 需要安排的次数（如 3 天餐饮）
    required_min_quality: float
    desc: str                 # 描述模板（展开时自动编号）


@dataclass(frozen=True)
class ScenarioCatalog:
    """一个消费场景的完整目录：环节模板 + 全部可选项。"""

    name: str
    description: str
    plan: tuple[PlanStepSpec, ...]
    options: tuple[Service, ...]                       # 基准价选项（Service 复用）
    failure_rates: dict[str, float] = field(default_factory=dict)  # 选项售罄率


def load_catalog(path: str | Path = _DEFAULT_CATALOG_PATH) -> ScenarioCatalog:
    """从 YAML 加载场景目录（无新增第三方依赖，pyyaml 已在 requirements 中）。"""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("场景扩展包需要 pyyaml（requirements.txt 已包含）") from e

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    plan = tuple(
        PlanStepSpec(
            task_type=str(step["type"]),
            count=int(step.get("count", 1)),
            required_min_quality=float(step["required_min_quality"]),
            desc=str(step["desc"]),
        )
        for step in data["plan"]
    )
    options, rates = [], {}
    for opt in data["options"]:
        svc = Service(
            name=str(opt["name"]),
            service_type=str(opt["category"]),
            cost_per_call=float(opt["base_price"]),
            quality_score=float(opt["quality"]),
            description=str(opt["description"]),
        )
        options.append(svc)
        rates[svc.name] = float(opt.get("failure_probability", 0.0))
    return ScenarioCatalog(
        name=str(data["scenario"]["name"]),
        description=str(data["scenario"].get("description", "")),
        plan=plan,
        options=tuple(options),
        failure_rates=rates,
    )


# ---------- 实时报价单 ----------

def build_price_registry(
    catalog: ScenarioCatalog, seed: int = 0,
) -> tuple[ServiceRegistry, dict[str, float]]:
    """按 seed 生成"本次查询到的报价单"。

    对每个选项：报价 = 基准价 × U(0.85, 1.15)（免费选项保持 0）。
    返回的注册表同时供 分解器估算 / 选择器决策 / 预订扣费 使用——
    三方一致，保证 spent ≤ total_budget 不变量。
    """
    rng = random.Random(seed)
    registry = ServiceRegistry(services=[])  # 空注册表：只装本场景选项，不加载默认 8 服务
    for base in catalog.options:
        if base.cost_per_call > 0:
            price = round(base.cost_per_call * rng.uniform(PRICE_NOISE_LOW, PRICE_NOISE_HIGH), 2)
        else:
            price = 0.0
        # 用 dataclasses.replace 复制选项并改写报价（不污染基准目录）
        quoted = Service(
            name=base.name,
            service_type=base.service_type,
            cost_per_call=price,
            quality_score=base.quality_score,
            description=base.description,
        )
        registry.register(quoted)
    return registry, dict(catalog.failure_rates)


# ---------- 场景分解器（模板驱动，与主场景 TaskPlanner 接口一致） ----------

class ScenarioPlanner:
    """按目录中的 plan 模板把消费任务展开为子任务序列。

    接口与 TaskPlanner.decompose(task) -> list[Subtask] 一致，
    可直接注入 AgentOrchestrator(planner=...)。
    estimated_cost 用报价单中该类别最低价做确定性估算（不臆测）。
    """

    def __init__(self, catalog: ScenarioCatalog, registry: ServiceRegistry) -> None:
        self._catalog = catalog
        self._registry = registry

    def decompose(self, task: str) -> list[Subtask]:
        subtasks: list[Subtask] = []
        order = 0
        for spec in self._catalog.plan:
            min_cost = min(
                (s.cost_per_call for s in self._registry.list_by_type(spec.task_type)),
                default=0.0,
            )
            for i in range(1, spec.count + 1):
                order += 1
                desc = spec.desc if spec.count == 1 else f"{spec.desc}（第{i}/{spec.count}）"
                subtasks.append(Subtask(
                    order=order,
                    task_type=spec.task_type,
                    required_min_quality=spec.required_min_quality,
                    estimated_cost=min_cost,
                    desc=desc,
                ))
        return subtasks


# ---------- 预订执行器（与 SimulatedServiceRunner 接口一致） ----------

_FAIL_REASONS = ("已满房", "已售罄", "报价已失效", "库存不足")


class BookingRunner:
    """把"执行服务"语义换成"安排行程选项"的模拟执行器。

    本项目不接入任何真实支付：所谓"扣费"只是把该选项的报价计入模拟账本，
    用于评估预算策略，不产生任何真实交易。

    - 安排成功：计入该选项本次报价（cost = service.cost_per_call），
      产出方案说明 + 体验描述；
    - 安排失败（按售罄率随机，模拟"方案不可用"）：不计费（cost=0.0）；
    - 预算不足时选择器会在调用前终止，因此不会出现"先花钱后失败"；
    - 与 BudgetLedger 解耦：记账由上层 AgentOrchestrator 完成。
    """

    def __init__(self, failure_rates: dict[str, float], seed: int = 0) -> None:
        self._failure_rates = failure_rates
        self._rng = random.Random(seed)

    def run(self, service: Service, subtask: Subtask) -> ServiceResult:
        rate = self._failure_rates.get(service.name, 0.0)
        if self._rng.random() < rate:
            reason = _FAIL_REASONS[self._rng.randrange(len(_FAIL_REASONS))]
            return ServiceResult(
                success=False,
                output="",
                cost=0.0,
                quality=0.0,
                error_msg=f"{service.name} {reason}",
            )
        return ServiceResult(
            success=True,
            output=(
                f"已安排「{service.name}」[{service.service_type}]，"
                f"预估价 ${service.cost_per_call:.2f}（模拟核算，无真实支付），"
                f"体验评分 {service.quality_score:.2f}。"
                f"{service.description}"
            ),
            cost=service.cost_per_call,
            quality=service.quality_score,
            error_msg="",
        )


# ---------- 场景一键装配 ----------

def build_travel_agent(seed: int = 0, catalog_path: str | Path = _DEFAULT_CATALOG_PATH):
    """装配旅行场景完整智能体（全部组件注入现有 AgentOrchestrator，零引擎修改）。

    返回 (agent, registry)——registry 为本次报价单，demo/前端可据此展示"查询到的价格"。
    """
    from ..agent_core import AgentOrchestrator
    from ..replanner import Replanner
    from ..service_selector import BudgetAwareSelector

    catalog = load_catalog(catalog_path)
    registry, rates = build_price_registry(catalog, seed=seed)
    agent = AgentOrchestrator(
        planner=ScenarioPlanner(catalog, registry),
        selector=BudgetAwareSelector(registry),
        runner=BookingRunner(rates, seed=seed),
        replanner=Replanner(registry),
        registry=registry,
    )
    return agent, registry
