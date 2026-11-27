"""Demo：循环调用各模拟服务，观察随机成功/失败。

失败率自动从 configs/service_config.yaml 加载（proposal 3.2）。
本 demo 不涉及账本记账，仅观察执行层行为。

运行方式（项目根目录下）：
    python demos/demo_sim_service.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.service_registry import ServiceRegistry
from src.simulated_services import SimulatedServiceRunner
from src.task_planner import Subtask

CALLS_PER_SERVICE = 10
SEED = 42  # 固定随机种子以便复现；改为 None 则每次运行结果不同


def main() -> None:
    registry = ServiceRegistry()
    runner = SimulatedServiceRunner(seed=SEED)

    print(f"每个服务模拟调用 {CALLS_PER_SERVICE} 次（o=成功, X=失败），seed={SEED}")
    print("=" * 64)

    # 先展示一条成功时的模拟输出样例
    sample = runner.run(registry.get("basic_search"), None)
    if sample.success:
        print(f"模拟输出样例: {sample.output}\n")

    for service in registry.list_all():
        subtask = Subtask(
            order=1,
            task_type=service.service_type,
            required_min_quality=0.0,
            estimated_cost=service.cost_per_call,
            desc="demo 测试子任务",
        )
        marks: list[str] = []
        fail_info: list[str] = []
        spent = 0.0
        for i in range(1, CALLS_PER_SERVICE + 1):
            r = runner.run(service, subtask)
            marks.append("o" if r.success else "X")
            spent += r.cost  # 仅统计展示，不写账本
            if not r.success:
                fail_info.append(f"第{i}次({r.error_msg})")
        p = runner.failure_probability(service.name)
        line = (
            f"{service.name:<16} [失败率 {p:.0%}] {''.join(marks)}  "
            f"成功 {marks.count('o')}/{CALLS_PER_SERVICE}，累计开销 ${spent:.2f}"
        )
        if fail_info:
            line += "  失败: " + "; ".join(fail_info)
        print(line)


if __name__ == "__main__":
    main()
