"""Service registry: maintain all available external services.

服务注册表：集中维护所有可用服务的元数据
（名称、单次调用价格、质量分、服务类别、描述）。

内置默认服务来源于 docs/proposal.md 3.2 节的模拟服务表。
注意：failure_probability 属于后续 simulated_services.py（随机失败模拟）的职责，
本阶段注册表不引入该字段。

后续任务分解器、服务选择器、模拟服务环境统一从本注册表查询服务。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Service:
    """单个可用服务的元数据（不可变）。"""

    name: str             # 服务唯一标识，如 "basic_search"
    service_type: str     # 服务类别：Search / Document Parsing / LLM Reasoning / Report Generation
    cost_per_call: float  # 单次调用价格（美元）
    quality_score: float  # 质量分，取值 0~1
    description: str      # 服务描述


# docs/proposal.md 3.2 节的模拟服务表（仅保留本阶段所需字段：价格与质量分）
DEFAULT_SERVICES: tuple[Service, ...] = (
    Service("basic_search",    "Search",            0.01, 0.70, "Low-cost basic web search service"),
    Service("premium_search",  "Search",            0.05, 0.90, "High-quality premium web search service"),
    Service("simple_parser",   "Document Parsing",  0.02, 0.60, "Low-cost simple document parser"),
    Service("advanced_parser", "Document Parsing",  0.08, 0.85, "High-quality advanced document parser"),
    Service("small_llm",       "LLM Reasoning",     0.10, 0.70, "Small LLM reasoning service"),
    Service("large_llm",       "LLM Reasoning",     0.50, 0.90, "Large LLM reasoning service"),
    Service("template_report", "Report Generation", 0.05, 0.65, "Template-based report generation"),
    Service("custom_report",   "Report Generation", 0.20, 0.90, "Customized high-quality report generation"),
)


class ServiceRegistry:
    """服务注册表：集中管理所有可用服务，支持按名称/类别查询。"""

    def __init__(self, services: list[Service] | None = None) -> None:
        """初始化注册表；不传 services 时加载 proposal 3.2 的内置默认服务。"""
        self._services: dict[str, Service] = {}
        for svc in (services if services is not None else DEFAULT_SERVICES):
            self.register(svc)

    def register(self, service: Service) -> None:
        """注册一个服务；同名服务会被覆盖。"""
        self._services[service.name] = service

    def get(self, name: str) -> Service:
        """按名称获取服务；不存在时抛出 KeyError。"""
        try:
            return self._services[name]
        except KeyError:
            raise KeyError(f"未知服务: {name}") from None

    def list_all(self) -> list[Service]:
        """返回全部服务（按注册顺序）。"""
        return list(self._services.values())

    def list_by_type(self, service_type: str) -> list[Service]:
        """按服务类别过滤，例如 "Search" 返回所有检索服务。"""
        return [s for s in self._services.values() if s.service_type == service_type]
