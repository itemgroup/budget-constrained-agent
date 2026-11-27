"""AuditorAgent: reconciliation + format validation + quality review.

审计员 Agent：交付前的独立验收，对应「核账 + 检查格式 + 输出合规格式」职责。

三件事按顺序做，前两件纯确定性（批量实验可复现、零 token），
只有第三件吃 LLM（可注入模拟 LLMJudge 或 RealLLMJudge）：

1. 对账（deterministic）：
   账本记录合计 == 已消耗；已消耗 + 剩余 == 总预算；未超支；
2. 格式校验（deterministic）：
   每个输出必须含 subtask_desc / task_type / service_name / output / quality，
   quality ∈ [0,1]，output 非空；
3. 质量评审（LLM）：
   completeness / correctness 打分（复用 evaluation.llm_judge）。

通过时产出结构化「合规交付物」；驳回时给出可读理由。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..evaluation.llm_judge import JudgeScore, LLMJudge
from .base import Agent

#: 交付物中每个子任务输出必须包含的字段
_REQUIRED_FIELDS = (
    "subtask_desc", "task_type", "service_name", "output", "quality",
)


@dataclass(frozen=True)
class AuditReport:
    """一次验收的完整结论。"""

    passed: bool                      # 是否通过
    reconciliation_ok: bool           # ① 对账是否一致
    reconciliation_detail: str        # ① 对账明细
    format_errors: list[str]          # ② 格式问题清单（空表示合规）
    score: JudgeScore | None          # ③ 质量评分（无产出时为 None）
    deliverable: str                  # 合规交付物（通过时为主产物）
    reason: str                       # 一句话结论 / 驳回理由


class AuditorAgent(Agent):
    """审计员：对账 + 格式校验 + 质量评审，产出 AuditReport。"""

    name = "AuditorAgent"

    def __init__(self, judge=None) -> None:
        """judge 需实现 score(task, outputs, planned_count) -> JudgeScore。

        缺省用模拟评分器 LLMJudge；真实环境注入 RealLLMJudge 即可。
        """
        self._judge = judge if judge is not None else LLMJudge()

    def act(self, blackboard) -> None:
        """执行验收并把 AuditReport 写入黑板。"""
        blackboard.audit_report = self.audit(blackboard)

    # ---------- 主流程 ----------

    def audit(self, blackboard) -> AuditReport:
        """完整验收：对账 -> 格式 -> 评分 -> 组交付物。"""
        recon_ok, recon_detail = self._reconcile(blackboard)
        format_errors = self._check_format(blackboard)

        score = None
        if blackboard.outputs:
            score = self._judge.score(
                blackboard.task, list(blackboard.outputs), len(blackboard.plan)
            )
            self._bill_llm(blackboard, calls=1)

        passed = recon_ok and not format_errors
        deliverable = self._build_deliverable(blackboard, score)

        if passed:
            tail = f"，Judge 综合 {score.overall:.2f}" if score else ""
            reason = f"验收通过：对账一致、格式合规{tail}"
        else:
            parts = []
            if not recon_ok:
                parts.append("账目不平")
            if format_errors:
                parts.append(f"{len(format_errors)} 处格式问题")
            reason = "验收驳回：" + "、".join(parts)

        self._log(blackboard, "audit", reason)
        return AuditReport(
            passed=passed,
            reconciliation_ok=recon_ok,
            reconciliation_detail=recon_detail,
            format_errors=format_errors,
            score=score,
            deliverable=deliverable,
            reason=reason,
        )

    # ---------- ① 对账 ----------

    def _reconcile(self, blackboard) -> tuple[bool, str]:
        ledger = blackboard.ledger
        record_total = round(sum(r.cost for r in ledger.records), 6)
        sum_ok = abs(record_total - ledger.spent) < 1e-6
        balance_ok = (
            abs(ledger.spent + ledger.remaining - ledger.total_budget) < 1e-6
        )
        no_overspend = ledger.spent <= ledger.total_budget + 1e-9
        detail = (
            f"记录合计 ${record_total:.4f} vs 已消耗 ${ledger.spent:.4f}"
            f"（{'一致' if sum_ok else '不一致'}）；"
            f"已消耗+剩余 ${ledger.spent + ledger.remaining:.4f} vs "
            f"总预算 ${ledger.total_budget:.4f}（{'平衡' if balance_ok else '不平衡'}）；"
            f"{'未超支' if no_overspend else '存在超支'}"
        )
        return (sum_ok and balance_ok and no_overspend), detail

    # ---------- ② 格式校验 ----------

    def _check_format(self, blackboard) -> list[str]:
        errors: list[str] = []
        for i, out in enumerate(blackboard.outputs, start=1):
            for field_name in _REQUIRED_FIELDS:
                if not hasattr(out, field_name):
                    errors.append(f"输出#{i} 缺少字段 {field_name}")
            quality = getattr(out, "quality", None)
            if not isinstance(quality, (int, float)) or not (
                0.0 <= float(quality) <= 1.0
            ):
                errors.append(
                    f"输出#{i} quality={quality} 越界（应在 0~1 之间）"
                )
            if not str(getattr(out, "output", "")).strip():
                errors.append(f"输出#{i} output 为空")
            if not str(getattr(out, "service_name", "")).strip():
                errors.append(f"输出#{i} service_name 为空")
        return errors

    # ---------- ③ 合规交付物 ----------

    def _build_deliverable(
        self, blackboard, score: JudgeScore | None
    ) -> str:
        ledger = blackboard.ledger
        lines = [
            f"# 交付物：{blackboard.task}",
            "",
            "## 预算执行",
            f"- 总预算 ${ledger.total_budget:.2f}｜已消耗 ${ledger.spent:.2f}"
            f"｜剩余 ${ledger.remaining:.2f}",
            f"- 子任务：计划 {len(blackboard.plan)} 项｜完成 "
            f"{len(blackboard.outputs)} 项｜跳过 {len(blackboard.skipped)} 项",
        ]
        if score is not None:
            lines.append(
                f"- LLM-Judge：完整性 {score.completeness:.2f}｜"
                f"正确性 {score.correctness:.2f}｜综合 {score.overall:.2f}"
            )
        lines.extend(["", "## 子任务结果"])
        if not blackboard.outputs:
            lines.append("- （本次运行没有产出）")
        for i, out in enumerate(blackboard.outputs, start=1):
            lines.append(
                f"{i}. [{out.task_type}] {out.subtask_desc}"
                f"（{out.service_name}，质量 {out.quality:.2f}）"
            )
            lines.append(f"   {out.output}")
        if blackboard.skipped:
            lines.extend(["", "## 缺损标记"])
            for desc in blackboard.skipped:
                lines.append(f"- 已跳过：{desc}")
        return "\n".join(lines)
