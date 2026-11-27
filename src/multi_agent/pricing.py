"""PricingAgent: produce the price sheet for this run.

询价员 Agent：负责产出「本次报价单」，写入黑板 price_sheet。

两种场景共用同一接口：
- 企业任务：取 ServiceRegistry 的基准价（proposal 3.2 的 8 个模拟服务）；
- 消费场景（行程等）：用 build_price_registry 生成 ±15% 波动的实时报价。

报价单由本 agent **独占写入**，选择器 / 执行器 / 账本三方共读同一份数据，
从而保证「决策价 == 扣费价」的一致性不变量。
本 agent 不参与决策、不扣费。
"""

from __future__ import annotations

from ..scenarios.scenario import build_price_registry
from ..service_registry import ServiceRegistry
from .base import Agent


class PricingAgent(Agent):
    """询价员：生成本次报价单。"""

    name = "PricingAgent"

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        catalog=None,
        seed: int = 0,
        label: str | None = None,
    ) -> None:
        """初始化。

        参数:
            registry: 已确定价目（企业场景的 8 服务，或行程场景已生成的报价单）
            catalog: 场景目录；传入时按 seed 现场生成波动报价（优先于 registry）
            seed: 报价波动随机种子（同 seed 得到同一张报价单，可复现）
            label: 报价来源说明（用于日志；缺省按企业/目录两种场景自动生成）
        """
        self._registry = registry
        self._catalog = catalog
        self._seed = seed
        self._label = label
        self.failure_rates: dict[str, float] = {}

    def act(self, blackboard) -> None:
        """生成本次报价单并写入黑板。"""
        if self._catalog is not None:
            registry, rates = build_price_registry(self._catalog, self._seed)
            self._registry = registry
            self.failure_rates = rates
            source = (
                f"场景目录「{self._catalog.name}」实时报价"
                f"（基准价 ±15% 波动，seed={self._seed}）"
            )
        else:
            if self._registry is None:
                self._registry = ServiceRegistry()
            source = self._label or "服务注册表基准价（proposal 3.2）"

        blackboard.registry = self._registry
        blackboard.price_sheet = {
            s.name: s.cost_per_call for s in self._registry.list_all()
        }
        self._log(
            blackboard, "quote",
            f"{source}：{len(blackboard.price_sheet)} 项报价写入黑板",
        )
