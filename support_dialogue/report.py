"""Read-only text and JSON reports.

The report layout follows a matter (客户事项), because that is what an agent
needs: what was asked, on which order, what was promised, which policy rule
was checked, where it stands, and the original sentence behind each fact.
"""

from __future__ import annotations

from typing import Optional

ESCALATION_STATUS = {
    "recommended": "建议升级",
    "in_progress": "处理中",
    "completed": "已完成",
    "needs_review": "需人工复核",
}

AUDIENCE = {
    "customer": "需问客户",
    "internal": "需内部确认",
}

PROMISE_STATUS = {
    "active": "当前有效",
    "adjusted": "已被后续口头调整",
    "repeated": "重复表述",
    "fulfilled": "已履行",
}

from .models import DialogueLedger, Evidence, Matter, Promise, Request

MAX_QUOTE = 80


def format_evidence(evidence: Optional[Evidence]) -> str:
    if evidence is None:
        return "（无原话）"
    label = evidence.speaker_label or {
        "agent": "客服",
        "customer": "客户",
    }.get(evidence.speaker, "发言人未标注")
    stamp = f" @{evidence.timestamp}" if evidence.timestamp else ""
    return (
        f"{evidence.record_id} 第{evidence.turn}轮 {incident_channel(evidence.channel)}"
        f"{stamp} {label}：“{clip(evidence.quote)}”"
    )


def incident_channel(channel: str) -> str:
    return {"web": "网页", "email": "邮件", "phone": "电话"}.get(channel, channel)


def clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_QUOTE else text[: MAX_QUOTE - 1] + "…"


def render_text(ledger: DialogueLedger) -> str:
    lines: list[str] = []
    lines.append(f"对话整理：{ledger.conversation_id}")
    lines.append(f"轮次数：{len(ledger.turns)}；客户事项：{len(ledger.matters)}")
    if ledger.as_of:
        lines.append(f"政策时点：{ledger.as_of}")
    lines.append("")

    for matter in ledger.matters:
        lines.extend(_render_matter(ledger, matter))
        lines.append("")

    if ledger.clarifications:
        lines.append("需要坐席追问")
        for clarification in ledger.clarifications:
            state = "待追问" if clarification.status == "open" else f"已澄清（{clarification.resolution}）"
            lines.append(
                f"- [{state}][{AUDIENCE.get(clarification.audience, clarification.audience)}] "
            f"{clarification.clarification_id}：{clarification.question}"
            )
            lines.append(f"  原因：{clarification.reason}")
            for evidence in clarification.evidence:
                lines.append(f"  原话：{format_evidence(evidence)}")
        lines.append("")

    if ledger.escalations:
        lines.append("升级进度")
        for escalation in ledger.escalations:
            lines.append(
                f"- {escalation.escalation_id} → {escalation.team_label}"
                f"（{ESCALATION_STATUS.get(escalation.status, escalation.status)}，"
                f"更新于第{escalation.updated_turn}轮）"
            )
            lines.append(f"  {escalation.reason}")
            if escalation.review_note:
                lines.append(f"  复核提示：{escalation.review_note}")
            for evidence in escalation.evidence:
                lines.append(f"  原话：{format_evidence(evidence)}")
        lines.append("")

    if ledger.notes:
        lines.append("整理说明")
        for note in ledger.notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("说明：本报告只读，不会发送任何客户答复。")
    return "\n".join(lines).rstrip() + "\n"


def _render_matter(ledger: DialogueLedger, matter: Matter) -> list[str]:
    lines = [
        f"客户事项 {matter.matter_id}｜{matter.label}",
        f"  关联订单：{('、'.join('订单 ' + item for item in matter.order_ids)) or '未确定'}",
        f"  状态：{matter.status_label}",
        f"  下一步：{matter.next_action}",
    ]
    for request_id in matter.request_ids:
        request = ledger.request(request_id)
        lines.extend(_render_request(ledger, request))
    # Promises the agent made before any request could be tied to them still
    # need to be visible to the next shift.
    request_promise_ids = {
        promise.promise_id
        for promise in ledger.promises
        if promise.request_id in matter.request_ids
    }
    for promise_id in matter.promise_ids:
        if promise_id in request_promise_ids:
            continue
        promise = ledger.promise(promise_id)
        lines.append(f"  · 未归属诉求的承诺 {promise.promise_id}")
        lines.extend(_render_promise(promise))
    return lines


def _render_request(ledger: DialogueLedger, request: Request) -> list[str]:
    lines = [
        f"  · 诉求 {request.request_id}（{request.intent_label}，{request.status_label}）",
        f"    订单：{('订单 ' + request.order_id) if request.order_id else '未确定'}",
    ]
    for evidence in request.evidence:
        lines.append(f"    原话：{format_evidence(evidence)}")
    if request.associations:
        association = request.associations[-1]
        if association.note:
            lines.append(f"    归属依据：{association.note}")
    if request.dissatisfaction:
        lines.append("    备注：原话含不满表达；不满本身不自动等同于赔付或退款请求。")
    if request.status_evidence is not None:
        lines.append(f"    处理结果原话：{format_evidence(request.status_evidence)}")

    active = ledger.active_assessment(request)
    if active is not None:
        lines.append(f"    政策依据：{active.conclusion}")
        lines.append(f"    政策原文：{active.policy_evidence[0].quote if active.policy_evidence else '（无）'}")
        for result in active.conditions:
            lines.append(f"      - {result.detail}")
        if active.trigger == "order_updated":
            lines.append("    说明：订单信息补正后已重新核对；此前的核对记录保留在历史中。")
        if active.order_evidence:
            for evidence in active.order_evidence:
                lines.append(f"    补正原话：{format_evidence(evidence)}")
        previous = [
            item
            for item in request.policy_assessments
            if item.assessment_id != active.assessment_id
        ]
        for item in previous:
            lines.append(
                f"    历史核对：{item.assessment_id}（{item.trigger}，"
                f"{item.policy_id}@{item.policy_version}）→ {item.eligible}"
            )

    for promise in ledger.promises:
        if promise.request_id != request.request_id:
            continue
        lines.extend(_render_promise(promise))

    if request.escalation_id:
        escalation = next(
            item for item in ledger.escalations if item.escalation_id == request.escalation_id
        )
        lines.append(
            f"    升级：{escalation.team_label}（{ESCALATION_STATUS.get(escalation.status, escalation.status)}，"
            f"更新于第{escalation.updated_turn}轮）"
        )
        if escalation.review_note:
            lines.append(f"      复核提示：{escalation.review_note}")

    for clarification_id in request.clarification_ids:
        clarification = next(
            item for item in ledger.clarifications if item.clarification_id == clarification_id
        )
        if clarification.status == "open":
            lines.append(
                f"    {AUDIENCE.get(clarification.audience, '需追问')}：{clarification.question}"
            )
    return lines


def _render_promise(promise: Promise) -> list[str]:
    stamp = promise.confirmed_at or f"第{promise.confirmed_turn}轮"
    lines = [
        f"    承诺 {promise.promise_id}（{PROMISE_STATUS.get(promise.status, promise.status)}）：{promise.quote}",
        f"      确认于 {stamp}；原话：{format_evidence(promise.evidence)}",
    ]
    if promise.deadline_text:
        lines.append(f"      承诺时限：{promise.deadline_text}（不得因新政策静默改写）")
    if promise.policy_version_at_confirmation:
        lines.append(f"      确认时适用政策：{promise.policy_version_at_confirmation}")
    if promise.duplicate_of_promise_id:
        lines.append(f"      重复自：{promise.duplicate_of_promise_id}")
    if promise.superseded_by_promise_id:
        lines.append(f"      后续调整记录：{promise.superseded_by_promise_id}")
    return lines


def render_matter(ledger: DialogueLedger, matter_id: str) -> str:
    matter = ledger.matter(matter_id)
    return "\n".join(_render_matter(ledger, matter)) + "\n"
