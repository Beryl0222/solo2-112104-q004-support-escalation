"""Deterministic dialogue analysis.

The analyser walks turns in order and builds a small ledger.  It is not a
general-purpose NLP system; it is a conservative, auditable pipeline for
synthetic support conversations where being wrong about "which order" is worse
than asking a question.

Key rules encoded here
----------------------
* A request is attached to an order explicitly, by a resolved anaphor, or not
  at all (then a clarification is raised).
* An agent commitment is frozen with its quote and confirmation turn.  Later
  utterances may *restate* or *adjust* it; they never edit it.
* Corrections bind future iterations.  When order facts change, every earlier
  policy assessment stays in the history and an unhandled request gets a fresh
  assessment.
* Dissatisfaction alone is not a refund request; a policy that permits refunds
  still has to pass each order condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

from .ingest import (
    collect_agent_labels,
    evidence_for_turn,
    load_turns,
    make_evidence,
)
from .models import (
    Clarification,
    Correction,
    DialogueLedger,
    Escalation,
    Evidence,
    Matter,
    OrderAssociation,
    OrderFieldUpdate,
    OrderView,
    Policy,
    PolicyAssessment,
    PolicyEvidence,
    ProgressEvent,
    Promise,
    Request,
    Turn,
)
from .policy import evaluate_order, load_policies, select_policy
from .text import (
    annotations,
    clause_has_claim,
    detect_intents,
    find_anaphor,
    find_deadline,
    find_order_ids,
    finds_completion,
    has_dissatisfaction,
    is_agent_utterance,
    is_question,
    looks_like_commitment,
    mentions_order,
    normalize_text,
    remove_annotations,
    short_quote,
    split_clauses,
    unique,
)

INTENT_LABELS = {
    "logistics": "物流查询",
    "refund": "退款/赔付申请",
    "cancel": "取消订单",
    "complaint": "投诉/不满",
    "unclear": "待确认诉求",
}

STATUS_LABELS = {
    "open": "待处理",
    "promised": "已承诺待履行",
    "completed": "已完成",
    "escalated": "升级处理中",
    "escalation_recommended": "建议升级",
    "needs_review": "需人工复核",
}

MATTER_LABELS = {
    "logistics": "物流事项",
    "refund": "退款事项",
    "cancel": "取消事项",
    "complaint": "投诉事项",
    "mixed": "混合事项",
    "unclear": "待归类事项",
}

TEAM_LABELS = {
    "logistics": "物流承运团队",
    "refund": "退款审核团队",
    "complaint": "客户关怀团队",
    "supervisor": "当班主管",
}

# A request whose policy check requires a team should be escalated there when
# no direct action is permitted, or when the customer explicitly complains.
ESCALATION_MATRIX = {
    "refund": "refund",
    "logistics": "logistics",
    "complaint": "complaint",
    "cancel": "supervisor",
}

ORDER_UPDATE_CUES = (
    "实际", "更正", "补正", "更新", "信息：", "现为", "应该是", "已改为",
    "属", "为", "金额是", "金额为", "签收", "承运", "付款", "购买", "下单",
)
_REFUND_WORDS = ("退款", "退回", "退钱", "赔付", "赔偿", "退款申请", "退款诉求")
_LOGISTICS_WORDS = ("物流", "快递", "承运", "配送", "运单", "包裹", "查询物流")


@dataclass
class _MutableRequest:
    request_id: str
    matter_id: str
    intent: str
    order_id: Optional[str]
    evidence: list[Evidence] = field(default_factory=list)
    correction_evidence: list[Evidence] = field(default_factory=list)
    associations: list[OrderAssociation] = field(default_factory=list)
    status: str = "open"
    status_evidence: Optional[Evidence] = None
    dissatisfaction: bool = False
    first_turn: int = 0
    last_turn: int = 0
    assessments: list[PolicyAssessment] = field(default_factory=list)
    active_assessment_id: Optional[str] = None
    clarification_ids: list[str] = field(default_factory=list)
    escalation_id: Optional[str] = None
    progress: list[ProgressEvent] = field(default_factory=list)

    def freeze(self) -> Request:
        return Request(
            request_id=self.request_id,
            matter_id=self.matter_id,
            intent=self.intent,
            intent_label=INTENT_LABELS.get(self.intent, self.intent),
            order_id=self.order_id,
            evidence=tuple(self.evidence),
            correction_evidence=tuple(self.correction_evidence),
            associations=tuple(self.associations),
            status=self.status,
            status_label=STATUS_LABELS.get(self.status, self.status),
            status_evidence=self.status_evidence,
            dissatisfaction=self.dissatisfaction,
            first_turn=self.first_turn,
            last_turn=self.last_turn,
            policy_assessments=tuple(self.assessments),
            active_assessment_id=self.active_assessment_id,
            clarification_ids=tuple(unique(self.clarification_ids)),
            escalation_id=self.escalation_id,
            progress=tuple(self.progress),
        )


@dataclass
class _MutablePromise:
    promise_id: str
    matter_id: Optional[str]
    request_id: Optional[str]
    intent: Optional[str]
    order_id: Optional[str]
    quote: str
    evidence: Evidence
    confirmed_turn: int
    confirmed_at: Optional[str]
    deadline_text: Optional[str]
    policy_version_at_confirmation: Optional[str]
    status: str = "active"
    duplicate_of_promise_id: Optional[str] = None
    superseded_by_promise_id: Optional[str] = None

    def freeze(self) -> Promise:
        return Promise(
            promise_id=self.promise_id,
            matter_id=self.matter_id,
            request_id=self.request_id,
            intent=self.intent,
            intent_label=INTENT_LABELS.get(self.intent or "", self.intent),
            order_id=self.order_id,
            quote=self.quote,
            evidence=self.evidence,
            confirmed_turn=self.confirmed_turn,
            confirmed_at=self.confirmed_at,
            deadline_text=self.deadline_text,
            policy_version_at_confirmation=self.policy_version_at_confirmation,
            duplicate_of_promise_id=self.duplicate_of_promise_id,
            superseded_by_promise_id=self.superseded_by_promise_id,
            status=self.status,
        )


class ConversationAnalyser:
    """Reusable, deterministic analyser for one synthetic conversation."""

    def __init__(
        self,
        turns: Sequence[Turn],
        policies: Sequence[Policy] = (),
        orders: Optional[Mapping[str, Mapping[str, Any]]] = None,
        as_of: Optional[str] = None,
        conversation_id: str = "conversation",
    ) -> None:
        self.turns = sorted(turns, key=lambda item: (item.turn, item.record_id))
        self.policies = tuple(policies)
        self.orders: dict[str, dict[str, Any]] = {
            order_id: dict(fields) for order_id, fields in (orders or {}).items()
        }
        self.order_updates: list[OrderFieldUpdate] = []
        self._seen_orders: list[str] = list(self.orders)
        self.order_update_evidence: dict[str, list[Evidence]] = {}
        self.as_of = as_of
        self.conversation_id = conversation_id
        self.agent_labels = collect_agent_labels(self.turns)
        self.requests: list[_MutableRequest] = []
        self.promises: list[_MutablePromise] = []
        self.matters: dict[str, dict[str, Any]] = {}
        self.assessments: list[PolicyAssessment] = []
        self.clarifications: list[Clarification] = []
        self.corrections: list[Correction] = []
        self.escalations: list[Escalation] = []
        self.progress: list[ProgressEvent] = []
        self.notes: list[str] = [
            "只读整理，不发送客户答复。",
            "每条诉求、承诺、政策结论与升级均保留原话、轮次与引用。",
        ]
        self._active_matter: Optional[str] = None
        self._counters: dict[str, int] = {}
        self._clarification_keys: set[Tuple[str, str, str]] = set()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self) -> DialogueLedger:
        for turn in self.turns:
            self._process_turn(turn)
        self._finalize_matters()
        return self._build_ledger()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}-{self._counters[prefix]}"

    def _new_matter(self, intent: str, order_id: Optional[str], turn: Turn) -> str:
        matter_id = self._next_id("matter")
        self.matters[matter_id] = {
            "label": MATTER_LABELS.get(intent, "待归类事项"),
            "intent": intent,
            "order_ids": [order_id] if order_id else [],
            "created_turn": turn.turn,
            "request_ids": [],
            "promise_ids": [],
            "escalation_ids": [],
            "clarification_ids": [],
        }
        self._progress(
            turn=turn,
            kind="matter_opened",
            description=f"按{MATTER_LABELS.get(intent, intent)}建立客户事项",
            request_id=None,
            matter_id=matter_id,
        )
        return matter_id

    def _matter_for(self, intent: str, order_id: Optional[str], turn: Turn) -> str:
        # Reuse an existing matter only for the same order and same family of
        # intent.  This keeps "订单 A 物流" apart from "订单 B 退款" even when
        # they are mentioned in the same conversation.
        if order_id is not None:
            for matter_id in reversed(list(self.matters)):
                matter = self.matters[matter_id]
                if order_id in matter["order_ids"] and matter["intent"] == intent:
                    self._active_matter = matter_id
                    return matter_id
        if self._active_matter is not None:
            matter = self.matters[self._active_matter]
            if (
                order_id is None
                and not matter["order_ids"]
                and matter["intent"] == intent
            ):
                return self._active_matter
        matter_id = self._new_matter(intent, order_id, turn)
        self._active_matter = matter_id
        return matter_id

    def _progress(
        self,
        turn: Turn,
        kind: str,
        description: str,
        request_id: Optional[str],
        matter_id: Optional[str],
        evidence: Optional[Evidence] = None,
    ) -> ProgressEvent:
        event = ProgressEvent(
            event_id=self._next_id("event"),
            request_id=request_id,
            matter_id=matter_id,
            kind=kind,
            description=description,
            turn=turn.turn,
            timestamp=turn.timestamp,
            evidence=evidence,
        )
        self.progress.append(event)
        return event

    def _link_request_to_matter(self, request: _MutableRequest) -> None:
        matter = self.matters[request.matter_id]
        if request.request_id not in matter["request_ids"]:
            matter["request_ids"].append(request.request_id)
        if request.order_id and request.order_id not in matter["order_ids"]:
            matter["order_ids"].append(request.order_id)

    # ------------------------------------------------------------------
    # turn processing
    # ------------------------------------------------------------------
    def _process_turn(self, turn: Turn) -> None:
        clauses = split_clauses(turn.text)
        if not clauses:
            return
        agent = is_agent_utterance(turn.text, self.agent_labels)

        if agent:
            self._process_agent_turn(turn, clauses)
        else:
            self._process_customer_turn(turn, clauses)
        self._check_policy_arrivals(turn)

    # -- agent ----------------------------------------------------------
    def _process_agent_turn(self, turn: Turn, clauses: Sequence[Tuple[str, int, int]]) -> None:
        correction_clauses = [
            (clause, start, end)
            for clause, start, end in clauses
            if self._is_correction_statement(clause)
        ]
        if correction_clauses:
            for clause, start, end in correction_clauses:
                self._apply_correction(turn, clause, make_evidence(turn, start, end), agent=True)
            return

        evidence = evidence_for_turn(turn)
        body = remove_annotations(turn.text)

        if _is_escalation_progress(body):
            self._update_escalation(turn, body, evidence)

        if looks_like_commitment(body, self.agent_labels):
            # Quote the whole agent utterance: splitting a commitment at a
            # comma would lose its deadline and create phantom promises.
            self._record_promise(turn, body, evidence)
            return

        if finds_completion(body):
            self._record_completion(turn, body, evidence)
            return

        if mentions_order(body):
            self._progress(
                turn=turn,
                kind="agent_note",
                description=f"坐席提及订单信息：{short_quote(body)}",
                request_id=self._current_request_id(),
                matter_id=self._active_matter,
                evidence=evidence,
            )

    # -- customer -------------------------------------------------------
    def _process_customer_turn(self, turn: Turn, clauses: Sequence[Tuple[str, int, int]]) -> None:
        seen_explicit: Optional[str] = None
        correction_order: Optional[str] = None
        correction_intent: Optional[str] = None
        correction_affected: list[_MutableRequest] = []
        correction_evidence: Optional[Evidence] = None

        for clause, start, end in clauses:
            evidence = make_evidence(turn, start, end)
            clause_orders = find_order_ids(clause)

            if self._is_correction_statement(clause):
                correction_order = clause_orders[0] if clause_orders else correction_order
                correction_intent = _correction_intent(clause, annotations(clause), self.requests) or correction_intent
                affected = self._apply_correction(
                    turn,
                    clause,
                    evidence,
                    agent=False,
                    order_id=correction_order,
                    intent=correction_intent,
                    reassess=False,
                )
                correction_affected.extend(affected)
                correction_evidence = correction_evidence or evidence
                continue

            # Later clauses of the same message often continue the correction
            # ("……凭证已上传，发货后第 6 天").  They inherit the correction's
            # order unless they name a different one.
            if correction_order is not None and (
                not clause_orders or clause_orders[0] == correction_order
            ) and _clause_looks_like_fact(clause):
                affected = self._apply_correction(
                    turn,
                    clause,
                    evidence,
                    agent=False,
                    order_id=correction_order,
                    intent=correction_intent,
                    kind="order_update",
                    reassess=False,
                )
                correction_affected.extend(affected)
                correction_evidence = correction_evidence or evidence
                continue

            if clause_orders:
                seen_explicit = clause_orders[0]

            if self._record_customer_restatement(turn, clause, evidence, clause_orders):
                continue

            if finds_completion(clause) and not is_question(clause):
                if self._record_customer_result(turn, clause, evidence, clause_orders):
                    continue

            intents = detect_intents(clause)
            if not intents:
                if has_dissatisfaction(clause):
                    self._record_dissatisfaction_without_ask(turn, clause, evidence, clause_orders)
                continue

            intent, cue = intents[0]
            if not clause_has_claim(clause) and not clause_orders:
                if intent == "complaint":
                    self._record_dissatisfaction_without_ask(turn, clause, evidence, clause_orders)
                continue

            self._create_request(
                turn=turn,
                clause=clause,
                evidence=evidence,
                intent=intent,
                order_ids=clause_orders,
                seen_explicit=seen_explicit,
                dissatisfaction=has_dissatisfaction(clause),
            )
        if correction_affected:
            self._reassess_after_corrections(
                unique_requests(correction_affected), turn, correction_evidence or evidence
            )

    # ------------------------------------------------------------------
    # request creation and order binding
    # ------------------------------------------------------------------
    def _create_request(
        self,
        turn: Turn,
        clause: str,
        evidence: Evidence,
        intent: str,
        order_ids: Sequence[str],
        seen_explicit: Optional[str],
        dissatisfaction: bool,
    ) -> None:
        explicit = order_ids[0] if order_ids else None

        # A follow-up clause in the same message usually continues the order
        # named in an earlier clause ("订单 C ……，麻烦一起退款").
        if explicit is None and seen_explicit is None and not find_anaphor(clause):
            if self._matter_order(self._active_matter) and _likely_continuation(clause):
                explicit = self._matter_order(self._active_matter)
        if explicit is None and seen_explicit is not None and not find_anaphor(clause):
            explicit = seen_explicit

        order_id: Optional[str] = explicit
        association_method = "explicit" if explicit else "unknown"
        association_evidence: Optional[Evidence] = evidence
        note = ""

        if order_id is None:
            anaphor = find_anaphor(clause)
            if anaphor:
                candidates = (
                    [seen_explicit]
                    if seen_explicit is not None
                    else self._anaphor_candidates(clause, intent)
                )
                if len(candidates) == 1:
                    order_id = candidates[0]
                    association_method = "anaphora"
                    note = f"“{anaphor}”按上下文唯一指向订单 {order_id}"
                else:
                    association_method = "anaphora"
                    note = f"“{anaphor}”无法唯一确定订单"
                    order_id = None
            else:
                association_method = "unknown"
                note = "原句未指明订单"

        matter_id = self._matter_for(intent, order_id, turn)
        merged = self._merge_into_existing(
            matter_id,
            intent,
            order_id,
            evidence,
            turn,
            OrderAssociation(
                order_id=order_id,
                method=association_method,
                evidence=association_evidence,
                note=note,
            ),
        )
        if merged is not None:
            return
        request = _MutableRequest(
            request_id=self._next_id("request"),
            matter_id=matter_id,
            intent=intent,
            order_id=order_id,
            dissatisfaction=dissatisfaction,
            first_turn=turn.turn,
            last_turn=turn.turn,
        )
        self._remember_order(order_id)
        request.evidence.append(evidence)
        request.associations.append(
            OrderAssociation(
                order_id=order_id,
                method=association_method,
                evidence=association_evidence,
                note=note,
            )
        )
        request.progress.append(
            self._progress(
                turn=turn,
                kind="request_detected",
                description=f"识别到{INTENT_LABELS[intent]}诉求",
                request_id=request.request_id,
                matter_id=matter_id,
                evidence=evidence,
            )
        )
        self.requests.append(request)
        self._link_request_to_matter(request)

        if order_id is None:
            if association_method == "anaphora":
                self._add_clarification(
                    turn=turn,
                    request=request,
                    kind="ambiguous_reference",
                    question=(
                        f"“{find_anaphor(clause) or '该订单'}”具体指哪个订单？"
                        "请列出候选订单号，或引用前面提到的订单名称。"
                    ),
                    reason="代词/指代无法唯一绑定订单，不能直接归入某个事项。",
                    candidates=tuple(self._known_orders()),
                    evidence=(evidence,),
                )
            else:
                self._add_clarification(
                    turn=turn,
                    request=request,
                    kind="missing_order",
                    question="这条诉求对应哪个订单？请补充订单号。",
                    reason="原句没有订单号，也没有可解析的指代。",
                    candidates=tuple(self._known_orders()),
                    evidence=(evidence,),
                )
        else:
            self._assess_policy(request, turn, evidence, trigger="initial")

    def _merge_into_existing(
        self,
        matter_id: str,
        intent: str,
        order_id: Optional[str],
        evidence: Evidence,
        turn: Turn,
        association: OrderAssociation,
    ) -> Optional[_MutableRequest]:
        """Fold a repeated ask into the still-open request instead of cloning it.

        The association for this sentence is kept so every follow-up can still
        be traced back to the words that tied it to the order.
        """
        for request in reversed(self.requests):
            if request.matter_id != matter_id:
                continue
            if request.intent != intent or request.order_id != order_id:
                continue
            if request.status not in {"open", "promised", "escalated"}:
                continue
            request.evidence.append(evidence)
            request.associations.append(association)
            request.last_turn = max(request.last_turn, turn.turn)
            if has_dissatisfaction(evidence.quote):
                request.dissatisfaction = True
            return request
        return None

    def _known_orders(self) -> list[str]:
        return unique(self._seen_orders)

    def _remember_order(self, order_id: Optional[str]) -> None:
        if order_id and order_id not in self._seen_orders:
            self._seen_orders.append(order_id)

    def _matter_order(self, matter_id: Optional[str]) -> Optional[str]:
        if matter_id is None:
            return None
        ids = self.matters[matter_id]["order_ids"]
        return ids[0] if len(ids) == 1 else None

    def _anaphor_candidates(self, clause: str, intent: str) -> list[str]:
        """Resolve a pronoun only when the antecedent is unambiguous.

        Priority:
        1. an explicit order earlier in the *same* utterance;
        2. existing requests of the *same intent* that already have an order
           (a "那个退款" needs a refund antecedent, not just any active order);
        3. the only known order in the conversation;
        4. otherwise every candidate is returned, which forces a clarification.
        """
        same_turn = find_order_ids(clause)
        if len(unique(same_turn)) == 1:
            return same_turn
        prior_orders = unique(
            item.order_id
            for item in self.requests
            if item.intent == intent and item.order_id
        )
        if len(prior_orders) == 1:
            return prior_orders
        known = unique(self._known_orders())
        if len(known) == 1:
            return known
        return prior_orders or known

    def _current_request_id(self) -> Optional[str]:
        if not self.requests:
            return None
        return self.requests[-1].request_id

    def _current_request(self) -> Optional[_MutableRequest]:
        if not self.requests:
            return None
        return self.requests[-1]

    def _check_policy_arrivals(self, turn: Turn) -> None:
        """Re-check unhandled requests when a newer policy version is in force.

        Confirmed promises are deliberately skipped: they keep the version
        that was in force when the agent confirmed them.
        """
        time = self._time_for(turn)
        if not time:
            return
        for request in self.requests:
            if request.status not in {"open", "promised", "escalated"}:
                continue
            current = select_policy(self.policies, request.intent, time)
            if current is None:
                continue
            if request.active_assessment_id is None:
                active_version = None
            else:
                active = next(
                    item
                    for item in request.assessments
                    if item.assessment_id == request.active_assessment_id
                )
                active_version = active.policy_version
            if active_version == current.version:
                continue
            active_turn = (
                next(
                    item
                    for item in request.assessments
                    if item.assessment_id == request.active_assessment_id
                ).checked_turn
                if request.active_assessment_id
                else None
            )
            if active_turn == turn.turn:
                continue
            assessment = self._assess_policy(
                request, turn, evidence_for_turn(turn), trigger="policy_updated"
            )
            self._handle_reassessment(
                request, assessment, turn, evidence_for_turn(turn), origin="policy_updated"
            )

    def _update_escalation(self, turn: Turn, clause: str, evidence: Evidence) -> None:
        request = self._resolve_promise_request(clause)
        if request is None:
            return
        if request.escalation_id is None:
            # The agent states a hand-off.  Record it even when policy data is
            # still incomplete; the status is "in progress", not "approved".
            self._create_escalation(
                request,
                turn,
                evidence,
                status="in_progress",
                reason=f"坐席在对话中明确说明已转交：{short_quote(clause)}",
            )
            return
        for index, escalation in enumerate(self.escalations):
            if escalation.escalation_id != request.escalation_id:
                continue
            if escalation.status == "completed":
                return
            if escalation.status == "needs_review" and "已完成" not in clause:
                # Do not let a routine hand-off note clear a policy review
                # flag.  Keep the review state; append the new evidence.
                self.escalations[index] = replace(
                    escalation,
                    updated_turn=turn.turn,
                    evidence=tuple(escalation.evidence) + (evidence,),
                )
                request.progress.append(
                    self._progress(
                        turn=turn,
                        kind="escalation_progress",
                        description=(
                            "收到升级进度说明，但该工单仍处于需人工复核状态："
                            + short_quote(clause)
                        ),
                        request_id=request.request_id,
                        matter_id=request.matter_id,
                        evidence=evidence,
                    )
                )
                return
            status = "completed" if "已完成" in clause else "in_progress"
            updated = replace(
                escalation,
                status=status,
                updated_turn=turn.turn,
                evidence=tuple(escalation.evidence) + (evidence,),
            )
            self.escalations[index] = updated
            request.progress.append(
                self._progress(
                    turn=turn,
                    kind="escalation_progress",
                    description=f"升级进度更新为“{status}”：{short_quote(clause)}",
                    request_id=request.request_id,
                    matter_id=request.matter_id,
                    evidence=evidence,
                )
            )
            return

    def _create_escalation(
        self,
        request: _MutableRequest,
        turn: Turn,
        evidence: Evidence,
        status: str,
        reason: str,
    ) -> Escalation:
        team = ESCALATION_MATRIX.get(request.intent, "supervisor")
        escalation = Escalation(
            escalation_id=self._next_id("escalation"),
            matter_id=request.matter_id,
            request_id=request.request_id,
            team=team,
            team_label=TEAM_LABELS.get(team, team),
            reason=reason,
            status=status,
            created_turn=turn.turn,
            updated_turn=turn.turn,
            evidence=(evidence,),
        )
        self.escalations.append(escalation)
        request.escalation_id = escalation.escalation_id
        self.matters[request.matter_id]["escalation_ids"].append(escalation.escalation_id)
        if status in {"in_progress", "completed"} and request.status not in {
            "completed",
            "escalated",
            "promised",
        }:
            # A recommendation is not progress: only an actual hand-off moves
            # the request into "escalated".
            request.status = "escalated"
        request.progress.append(
            self._progress(
                turn=turn,
                kind="escalation_recorded",
                description=f"{escalation.team_label}升级状态：{status}",
                request_id=request.request_id,
                matter_id=request.matter_id,
                evidence=evidence,
            )
        )
        return escalation

    # ------------------------------------------------------------------
    # promises
    # ------------------------------------------------------------------
    def _record_promise(self, turn: Turn, clause: str, evidence: Evidence) -> None:
        targets = self._promise_targets(clause)
        if not targets:
            targets = [None]
        deadline = find_deadline(remove_annotations(clause))
        allow_supersede = len(targets) == 1
        for target in targets:
            self._record_promise_for_target(
                turn, clause, evidence, target, deadline, allow_supersede
            )

    def _promise_targets(self, clause: str) -> list[Optional[_MutableRequest]]:
        """One commitment can cover one or several active matters.

        "两单今天下班前都给您核对" is a single utterance but two separate
        promises; each order keeps its own quote and confirmation time so a
        later shift can see exactly what was promised for which order.
        """
        open_requests = [
            item
            for item in self.requests
            if item.status in {"open", "promised", "escalated"}
        ]
        multi = any(
            marker in clause for marker in ("两单", "两个订单", "两个都", "都给您", "分别")
        )
        if multi:
            by_order: dict[str, _MutableRequest] = {}
            for request in open_requests:
                if request.order_id and request.order_id not in by_order:
                    by_order[request.order_id] = request
            if len(by_order) >= 2:
                return list(by_order.values())
        request = self._resolve_promise_request(clause)
        return [request]

    def _record_promise_for_target(
        self,
        turn: Turn,
        clause: str,
        evidence: Evidence,
        request: Optional[_MutableRequest],
        deadline: Optional[str],
        allow_supersede: bool = True,
    ) -> None:
        intent = request.intent if request else _intent_from_clause(clause)
        order_id = request.order_id if request else _order_from_clause(clause)
        matter_id = request.matter_id if request else self._active_matter
        policy_version = self._policy_version_for(intent, turn)

        existing = self._matching_promise(clause, order_id, intent) if allow_supersede else None
        if existing is not None:
            if _same_commitment(existing, clause):
                repeat = _MutablePromise(
                    promise_id=self._next_id("promise"),
                    matter_id=matter_id,
                    request_id=request.request_id if request else None,
                    intent=intent,
                    order_id=order_id,
                    quote=clause,
                    evidence=evidence,
                    confirmed_turn=turn.turn,
                    confirmed_at=turn.timestamp,
                    deadline_text=deadline,
                    policy_version_at_confirmation=policy_version,
                    status="repeated",
                    duplicate_of_promise_id=existing.promise_id,
                )
                self.promises.append(repeat)
                if matter_id:
                    self.matters[matter_id]["promise_ids"].append(repeat.promise_id)
                self._progress(
                    turn=turn,
                    kind="promise_repeated",
                    description=(
                        f"坐席重复了已有承诺 {existing.promise_id}；"
                        "以第一条承诺的确认时间为准，不重复计入新承诺。"
                    ),
                    request_id=existing.request_id,
                    matter_id=existing.matter_id,
                    evidence=evidence,
                )
                if request is not None:
                    if request.status == "open":
                        request.status = "promised"
                    request.last_turn = max(request.last_turn, turn.turn)
                return
            # Same order + same intent but a different commitment: record a
            # new promise that explicitly says it adjusts the earlier one.
            promise_id = self._next_id("promise")
            superseded = existing
            superseded.status = "adjusted"
            superseded.superseded_by_promise_id = promise_id
        else:
            promise_id = self._next_id("promise")

        promise = _MutablePromise(
            promise_id=promise_id,
            matter_id=matter_id,
            request_id=request.request_id if request else None,
            intent=intent,
            order_id=order_id,
            quote=clause,
            evidence=evidence,
            confirmed_turn=turn.turn,
            confirmed_at=turn.timestamp,
            deadline_text=deadline,
            policy_version_at_confirmation=policy_version,
        )
        self.promises.append(promise)
        if matter_id:
            self.matters[matter_id]["promise_ids"].append(promise_id)
        if request is not None:
            if request.status == "open":
                request.status = "promised"
            request.last_turn = max(request.last_turn, turn.turn)
            request.progress.append(
                self._progress(
                    turn=turn,
                    kind="promise_recorded",
                    description=f"记录已确认承诺 {promise_id} 原文",
                    request_id=request.request_id,
                    matter_id=matter_id,
                    evidence=evidence,
                )
            )

    def _resolve_promise_request(self, clause: str) -> Optional[_MutableRequest]:
        """Attach an agent commitment to the request it is actually about.

        An agent may promise something about "订单 A" while the last detected
        request was about "订单 B".  Explicit order/action cues win; otherwise
        the most recent request is used.
        """
        explicit_order = _order_from_clause(clause)
        clause_intent = _intent_from_clause(clause)
        candidates = [
            item
            for item in reversed(self.requests)
            if item.status in {"open", "promised", "escalated"}
        ]
        if explicit_order is not None:
            ordered = [item for item in candidates if item.order_id == explicit_order]
            if clause_intent:
                exact = [item for item in ordered if item.intent == clause_intent]
                if exact:
                    return exact[0]
            if ordered:
                return ordered[0]
        if clause_intent:
            matching = [item for item in candidates if item.intent == clause_intent]
            if matching:
                return matching[0]
        return candidates[0] if candidates else None

    def _matching_promise(
        self, clause: str, order_id: Optional[str], intent: Optional[str]
    ) -> Optional[_MutablePromise]:
        candidates = [
            promise
            for promise in self.promises
            if promise.status in {"active", "adjusted"}
            and promise.order_id == order_id
            and (not intent or promise.intent == intent)
        ]
        # An exact repeat of an earlier commitment (possibly several turns
        # later) must be recognised as a repeat, not as a new promise.  This
        # is exactly the hand-over risk: a new shift re-promising the same
        # deadline that an earlier shift already committed.
        for promise in reversed(candidates):
            if _same_commitment(promise, clause):
                return promise
        return candidates[-1] if candidates else None
        return None

    def _policy_version_for(self, intent: Optional[str], turn: Turn) -> Optional[str]:
        if not intent:
            return None
        policy = select_policy(self.policies, intent, self._time_for(turn))
        return f"{policy.policy_id}@{policy.version}" if policy else None

    def _time_for(self, turn: Turn) -> Optional[str]:
        return turn.timestamp or self.as_of

    # ------------------------------------------------------------------
    # completion / dissatisfaction
    # ------------------------------------------------------------------
    def _record_completion(self, turn: Turn, clause: str, evidence: Evidence) -> None:
        request = self._current_request()
        if request is None:
            return
        self._mark_complete(request, turn, evidence, f"坐席报告处理结果：{short_quote(clause)}")

    def _record_customer_restatement(
        self,
        turn: Turn,
        clause: str,
        evidence: Evidence,
        order_ids: Sequence[str],
    ) -> bool:
        """Note a customer echoing a commitment without creating a request.

        Hand-over notes often look like "明天上午给您，别重复了".  Treating that
        as a new customer request would let the next shift lose the fact that
        the time was the *agent's* commitment.
        """
        target_order = order_ids[0] if order_ids else None
        for promise in reversed(self.promises):
            if promise.status not in {"active", "adjusted", "repeated"}:
                continue
            if target_order is not None and promise.order_id != target_order:
                continue
            if not promise.deadline_text or promise.deadline_text not in clause:
                continue
            if _text_overlap(promise.quote, clause) < 0.5:
                continue
            self._progress(
                turn=turn,
                kind="customer_restated_promise",
                description=(
                    f"客户复述了承诺 {promise.promise_id} 的时限“{promise.deadline_text}”；"
                    "按原承诺记录处理，不另建诉求。"
                ),
                request_id=promise.request_id,
                matter_id=promise.matter_id,
                evidence=evidence,
            )
            return True
        return False

    def _record_customer_result(
        self,
        turn: Turn,
        clause: str,
        evidence: Evidence,
        order_ids: Sequence[str],
    ) -> bool:
        """Record a result reported by the customer, linked to a real request."""
        target_order = order_ids[0] if order_ids else None
        for request in reversed(self.requests):
            if request.status == "completed":
                continue
            if target_order and request.order_id != target_order:
                continue
            if target_order is None and request.order_id != self._matter_order(self._active_matter):
                continue
            if request.intent not in {"refund", "logistics", "cancel"}:
                continue
            self._mark_complete(request, turn, evidence, f"客户报告处理结果：{short_quote(clause)}")
            return True
        return False

    def _record_dissatisfaction_without_ask(
        self, turn: Turn, clause: str, evidence: Evidence, order_ids: Sequence[str]
    ) -> None:
        # Informational only.  A complaint is opened as its own request; a
        # refund request is NOT created from dissatisfaction alone.  Do not
        # inherit an order from unrelated context: an unbound complaint must
        # lead to "which order?" rather than silently attaching to the last one.
        order_id = order_ids[0] if order_ids else None
        matter_id = self._matter_for("complaint", order_id, turn)
        for existing in reversed(self.requests):
            if (
                existing.matter_id == matter_id
                and existing.intent == "complaint"
                and existing.order_id == order_id
                and existing.status == "open"
            ):
                existing.evidence.append(evidence)
                existing.dissatisfaction = True
                existing.last_turn = max(existing.last_turn, turn.turn)
                return
        request = _MutableRequest(
            request_id=self._next_id("request"),
            matter_id=matter_id,
            intent="complaint",
            order_id=order_id,
            dissatisfaction=True,
            first_turn=turn.turn,
            last_turn=turn.turn,
        )
        self._remember_order(order_id)
        request.evidence.append(evidence)
        request.associations.append(
            OrderAssociation(
                order_id=order_id,
                method="explicit" if order_ids else "unknown",
                evidence=evidence,
                note=(
                    "仅表达不满，未提出赔付/退款；不自动生成退款诉求"
                    if not order_ids
                    else "仅表达不满，未提出赔付/退款"
                ),
            )
        )
        self.requests.append(request)
        self._link_request_to_matter(request)
        request.progress.append(
            self._progress(
                turn=turn,
                kind="dissatisfaction_noted",
                description="记录不满，未将其等同为赔付请求",
                request_id=request.request_id,
                matter_id=matter_id,
                evidence=evidence,
            )
        )
        if order_id is None:
            self._add_clarification(
                turn=turn,
                request=request,
                kind="missing_order",
                question="客户表达不满，但未说明涉及哪个订单。请确认订单号后再决定是否升级。",
                reason="不满陈述没有订单指向，不能自动升级为赔付。",
                candidates=tuple(self._known_orders()),
                evidence=(evidence,),
            )

    # ------------------------------------------------------------------
    # corrections
    # ------------------------------------------------------------------
    def _is_correction_statement(self, clause: str) -> bool:
        if any(phrase in clause for phrase in ("补充：", "补充:", "更正：", "更正:")):
            return True
        if any(
            phrase in clause
            for phrase in (
                "提到的", "刚才的", "刚才说的", "上面说的", "此前说的",
                "我说的是", "指的是", "是指", "说的是",
            )
        ):
            return True
        return any(
            phrase in clause
            for phrase in ("实际是", "实际为", "已改为", "更正为", "应当为", "不是订单")
        )

    def _apply_correction(
        self,
        turn: Turn,
        clause: str,
        evidence: Evidence,
        agent: bool,
        order_id: Optional[str] = None,
        intent: Optional[str] = None,
        kind: Optional[str] = None,
        reassess: bool = True,
    ) -> list[_MutableRequest]:
        annotations_found = annotations(clause)
        order_ids = [order_id] if order_id else find_order_ids(clause)
        intent = intent or _correction_intent(clause, annotations_found, self.requests)
        fields = _extract_order_fields(clause)
        affected = self._affected_requests(intent, order_ids)

        kind = kind or ("order_update" if fields else "binding")
        note = ""
        if annotations_found:
            note = "括注指定了被说明的意图：" + "、".join(annotations_found)
        if not affected:
            note = (note + "；" if note else "") + "未找到需要重核的已识别诉求"

        correction = Correction(
            correction_id=self._next_id("correction"),
            kind=kind,
            turn=turn.turn,
            intent=intent,
            order_id=order_ids[0] if order_ids else None,
            fields=fields,
            affected_request_ids=tuple(unique([item.request_id for item in affected])),
            evidence=evidence,
            note=note,
        )
        self.corrections.append(correction)

        if order_ids:
            self._remember_order(order_ids[0])
        if fields and order_ids:
            order_id = order_ids[0]
            self.orders.setdefault(order_id, {}).update(fields)
            update = OrderFieldUpdate(
                update_id=self._next_id("update"),
                order_id=order_id,
                fields=fields,
                turn=turn.turn,
                evidence=evidence,
            )
            self.order_updates.append(update)
            self.order_update_evidence.setdefault(order_id, []).append(evidence)

        for request in affected:
            request.correction_evidence.append(evidence)
            self._progress(
                turn=turn,
                kind="correction_linked",
                description=(
                    f"修订关联到诉求 {request.request_id}（{INTENT_LABELS[request.intent]}）"
                    + (f"，意图重绑定为 {INTENT_LABELS[intent]}" if intent and intent != request.intent else "")
                ),
                request_id=request.request_id,
                matter_id=request.matter_id,
                evidence=evidence,
            )
            if intent and intent != request.intent:
                old = INTENT_LABELS[request.intent]
                request.intent = intent
                request.dissatisfaction = request.dissatisfaction or intent == "complaint"
                self._rebind_matter(request, turn, evidence)
                self._progress(
                    turn=turn,
                    kind="intent_rebound",
                    description=f"意图由“{old}”更正为“{INTENT_LABELS[intent]}”",
                    request_id=request.request_id,
                    matter_id=request.matter_id,
                    evidence=evidence,
                )
            elif request.intent == "unclear":
                request.intent = intent or "unclear"
                self._rebind_matter(request, turn, evidence)
            # Policy re-check is done once per turn after all fact clauses are
            # applied; earlier assessment objects are never overwritten.
            self._rebind_missing_order(request, order_ids, evidence)
            self._refresh_completed_request(request, turn, evidence)
        return affected

    def _reassess_after_corrections(
        self,
        requests: Sequence[_MutableRequest],
        turn: Turn,
        evidence: Evidence,
    ) -> None:
        """Run one fresh policy check per affected request after a correction.

        All fact fragments in the turn are applied first, so the assessment sees
        the complete corrected snapshot.  Earlier assessment objects remain in
        the history untouched.
        """
        for request in requests:
            if request.status not in {"open", "promised", "escalated"}:
                self._resolve_clarifications(request, turn, evidence)
                continue
            assessment = self._assess_policy(
                request, turn, evidence, trigger="order_updated"
            )
            self._handle_reassessment(
                request, assessment, turn, evidence, origin="order_updated"
            )

    def _handle_reassessment(
        self,
        request: _MutableRequest,
        assessment: Optional[PolicyAssessment],
        turn: Turn,
        evidence: Evidence,
        origin: str,
    ) -> None:
        """React to a fresh policy check without rewriting history.

        A negative result on new policy/order facts keeps every prior
        assessment and every confirmed promise.  An existing escalation is
        marked for human review; it is not silently re-labelled as approved.
        """
        if assessment is not None and assessment.eligible == "no":
            if request.escalation_id is not None:
                self._flag_escalation_review(
                    request, turn, evidence, assessment, origin=origin
                )
            if any(
                promise.request_id == request.request_id
                and promise.status in {"active", "adjusted"}
                for promise in self.promises
            ):
                self._progress(
                    turn=turn,
                    kind="promise_retained",
                    description=(
                        "新政策/补正资料下不再满足条件；已确认承诺保留原文和确认时间，"
                        "等待人工复核，不静默改写。"
                    ),
                    request_id=request.request_id,
                    matter_id=request.matter_id,
                    evidence=evidence,
                )
        self._resolve_clarifications(request, turn, evidence)

    def _flag_escalation_review(
        self,
        request: _MutableRequest,
        turn: Turn,
        evidence: Evidence,
        assessment: PolicyAssessment,
        origin: str,
    ) -> None:
        for index, escalation in enumerate(self.escalations):
            if escalation.escalation_id != request.escalation_id:
                continue
            if escalation.status == "completed":
                return
            if escalation.status != "needs_review":
                self.escalations[index] = replace(
                    escalation,
                    status="needs_review",
                    updated_turn=turn.turn,
                    evidence=tuple(escalation.evidence) + (evidence,),
                    review_note=(
                        f"第{turn.turn}轮核对 {assessment.assessment_id} 显示订单不再满足"
                        f"政策 {assessment.policy_id}@{assessment.policy_version}；"
                        "原升级理由保留，需人工复核后决定是否撤回或补充材料。"
                    ),
                )
            self._progress(
                turn=turn,
                kind="policy_conflict",
                description=(
                    f"核对 {assessment.assessment_id}（{origin}）显示订单不再满足"
                    f"政策 {assessment.policy_id}@{assessment.policy_version}；"
                    "已有升级转为需人工复核，已确认承诺不改写。"
                ),
                request_id=request.request_id,
                matter_id=request.matter_id,
                evidence=evidence,
            )
            return

    def _refresh_completed_request(
        self, request: _MutableRequest, turn: Turn, evidence: Evidence
    ) -> None:
        # A correction can also report completion ("已退回").  Treat it as a
        # result only when the clause says so, never inferred.
        if finds_completion(evidence.quote) and request.status != "completed":
            self._mark_complete(request, turn, evidence, "修订原句报告处理结果")

    def _mark_complete(
        self,
        request: _MutableRequest,
        turn: Turn,
        evidence: Evidence,
        description: str,
    ) -> None:
        request.status = "completed"
        request.status_evidence = evidence
        request.last_turn = max(request.last_turn, turn.turn)
        request.progress.append(
            self._progress(
                turn=turn,
                kind="completion_reported",
                description=description,
                request_id=request.request_id,
                matter_id=request.matter_id,
                evidence=evidence,
            )
        )
        for promise in self.promises:
            if promise.request_id == request.request_id and promise.status == "active":
                # Fulfilment is recorded on the promise record; its quote and
                # confirmation time are untouched.
                promise.status = "fulfilled"
        for index, clarification in enumerate(self.clarifications):
            if clarification.request_id != request.request_id or clarification.status != "open":
                continue
            self.clarifications[index] = replace(
                clarification,
                status="resolved",
                resolution="该诉求已由后续处理结果关闭，无需继续追问",
            )

    def _rebind_matter(
        self, request: _MutableRequest, turn: Turn, evidence: Evidence
    ) -> None:
        old_matter = request.matter_id
        new_matter = self._new_matter(request.intent, request.order_id, turn)
        request.matter_id = new_matter
        self.matters[new_matter]["request_ids"].append(request.request_id)
        if request.order_id and request.order_id not in self.matters[new_matter]["order_ids"]:
            self.matters[new_matter]["order_ids"].append(request.order_id)
        self.matters[old_matter]["request_ids"] = [
            item for item in self.matters[old_matter]["request_ids"] if item != request.request_id
        ]
        self._active_matter = new_matter
        self._progress(
            turn=turn,
            kind="matter_rebound",
            description=f"诉求从 {old_matter} 移入 {new_matter}",
            request_id=request.request_id,
            matter_id=new_matter,
            evidence=evidence,
        )

    def _affected_requests(
        self, intent: Optional[str], order_ids: Sequence[str]
    ) -> list[_MutableRequest]:
        # A correction always names the thing being corrected (an order, an
        # intent, or both).  Requests on other orders are never touched.
        target_order = order_ids[0] if order_ids else None
        affected: list[_MutableRequest] = []
        for request in self.requests:
            if target_order is not None and request.order_id not in {target_order, None}:
                continue
            if intent is not None and request.intent != intent:
                continue
            affected.append(request)
        if not affected and self.requests and intent is not None:
            # Last resort: the correction names an intent but no earlier
            # request matched exactly.  Only take requests with that intent
            # from the active matter, never unrelated ones.
            if self._active_matter:
                affected = [
                    item for item in self.requests
                    if item.matter_id == self._active_matter and item.intent == intent
                ]
        return affected

    def _rebind_missing_order(
        self,
        request: _MutableRequest,
        order_ids: Sequence[str],
        evidence: Evidence,
    ) -> None:
        if request.order_id is not None or not order_ids:
            return
        order_id = order_ids[0]
        request.order_id = order_id
        request.associations = tuple(request.associations) + (
            OrderAssociation(
                order_id=order_id,
                method="correction",
                evidence=evidence,
                note="后续补充信息明确了订单",
            ),
        )
        if order_id not in self.matters[request.matter_id]["order_ids"]:
            self.matters[request.matter_id]["order_ids"].append(order_id)

    def _resolve_clarifications(
        self, request: _MutableRequest, turn: Turn, evidence: Evidence
    ) -> None:
        active = (
            next(
                item
                for item in request.assessments
                if item.assessment_id == request.active_assessment_id
            )
            if request.active_assessment_id
            else None
        )
        missing_codes = (
            {item.condition.code for item in active.missing_conditions}
            if active
            else set()
        )
        for index, clarification in enumerate(self.clarifications):
            if clarification.request_id != request.request_id or clarification.status != "open":
                continue
            if clarification.kind in {"missing_order", "ambiguous_reference"} and request.order_id:
                self.clarifications[index] = replace(
                    clarification,
                    status="resolved",
                    resolution=f"已绑定订单 {request.order_id}",
                )
            elif clarification.kind == "missing_policy_condition" and (
                active is None
                or active.eligible != "unknown"
                or clarification.condition_code not in missing_codes
            ):
                self.clarifications[index] = replace(
                    clarification,
                    status="resolved",
                    resolution=f"第 {turn.turn} 轮补充资料后条件已可判断",
                )
            elif clarification.kind == "missing_policy_condition" and active is not None:
                # Still missing, but a newer policy version may now be in
                # force.  Refresh the quoted condition without closing it.
                current = next(
                    (
                        item
                        for item in active.missing_conditions
                        if item.condition.code == clarification.condition_code
                    ),
                    None,
                )
                if current is not None and current.source_evidence is not None:
                    self.clarifications[index] = replace(
                        clarification,
                        policy_evidence=(current.source_evidence,),
                        reason=(
                            f"政策 {active.policy_id}@{active.policy_version} "
                            f"的条件“{current.condition.description}”缺少订单资料，"
                            "暂不能判断该诉求是否符合政策条件。"
                        ),
                    )

    # ------------------------------------------------------------------
    # policy assessment
    # ------------------------------------------------------------------
    def _assess_policy(
        self,
        request: _MutableRequest,
        turn: Turn,
        evidence: Evidence,
        trigger: str,
    ) -> None:
        if request.intent == "complaint" and request.order_id is None:
            # A complaint cannot be matched against an order-level policy
            # before we know the order.  Asking "which order?" is the honest
            # conclusion; do not report "不满足条件".
            return None
        policy = select_policy(self.policies, request.intent, self._time_for(turn))
        if policy is None:
            return None
        order_view = self._order_view(request.order_id) if request.order_id else None
        extra_fields = {}
        if request.intent == "complaint":
            extra_fields["order_identified"] = request.order_id is not None
        evaluation = evaluate_order(policy, order_view, extra_fields=extra_fields)
        order_evidence = self._order_evidence(request.order_id)

        assessment_id = self._next_id("assessment")
        policy_evidence = self._policy_evidence(policy, evaluation)
        assessment = PolicyAssessment(
            assessment_id=assessment_id,
            request_id=request.request_id,
            matter_id=request.matter_id,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            policy_allows=policy.allows,
            eligible=evaluation.eligible,
            conclusion=evaluation.conclusion,
            conditions=evaluation.results,
            missing_conditions=evaluation.missing,
            checked_turn=turn.turn,
            checked_at=turn.timestamp,
            trigger=trigger,
            trigger_evidence=evidence,
            policy_evidence=policy_evidence,
            order_id=request.order_id,
            order_fields=dict(order_view.fields) if order_view else {},
            order_evidence=order_evidence,
            policy_summary=policy.summary,
        )
        request.assessments.append(assessment)
        request.active_assessment_id = assessment_id
        self.assessments.append(assessment)
        request.progress.append(
            self._progress(
                turn=turn,
                kind="policy_checked",
                description=f"政策核对 {assessment_id}（{trigger}）：{evaluation.conclusion}",
                request_id=request.request_id,
                matter_id=request.matter_id,
                evidence=evidence,
            )
        )

        if evaluation.eligible == "yes":
            self._maybe_escalate(request, turn, evidence, policy)
        elif evaluation.eligible == "unknown":
            self._clarify_missing_conditions(request, turn, evidence, evaluation)
        # "no" is a negative eligibility result: escalation is still reviewed
        # for refund disputes, but never automatically treated as approved.
        return assessment

    def _policy_evidence(self, policy: Policy, evaluation: Any) -> Tuple[PolicyEvidence, ...]:
        items = [
            PolicyEvidence(
                policy_id=policy.policy_id,
                version=policy.version,
                source=policy.source,
                quote=policy.summary,
            )
        ]
        for result in evaluation.results:
            if result.source_evidence is not None:
                items.append(result.source_evidence)
        return tuple(items)

    def _clarify_missing_conditions(
        self,
        request: _MutableRequest,
        turn: Turn,
        evidence: Evidence,
        evaluation: Any,
    ) -> None:
        for result in evaluation.missing:
            question = result.condition.prompt
            self._add_clarification(
                turn=turn,
                request=request,
                kind="missing_policy_condition",
                question=question,
                reason=(
                    f"政策 {evaluation.policy.policy_id}@{evaluation.policy.version} "
                    f"的条件“{result.condition.description}”缺少订单资料，"
                    "暂不能判断该诉求是否符合政策条件。"
                ),
                candidates=(),
                evidence=(evidence,),
                policy_evidence=(result.source_evidence,) if result.source_evidence else (),
                condition_code=result.condition.code,
                audience=result.condition.audience,
            )

    def _maybe_escalate(
        self,
        request: _MutableRequest,
        turn: Turn,
        evidence: Evidence,
        policy: Policy,
    ) -> None:
        # A refund is never sent onward just because the policy "allows" it:
        # the caller only reaches this method after every order condition is
        # met.  The agent still has to do a manual hand-off.
        team = ESCALATION_MATRIX.get(request.intent)
        if team is None:
            return
        if request.intent == "logistics" and request.status == "completed":
            return
        if request.escalation_id is not None:
            return
        self._create_escalation(
            request,
            turn,
            evidence,
            status="recommended",
            reason=(
                f"{INTENT_LABELS[request.intent]}满足政策条件，需人工转交"
                f"{TEAM_LABELS.get(team, team)}处理（不会自动发送客户答复）。"
            ),
        )

    def _order_view(self, order_id: Optional[str]) -> Optional[OrderView]:
        if not order_id:
            return None
        fields = self.orders.get(order_id, {})
        updates = tuple(
            item for item in self.order_updates if item.order_id == order_id
        )
        return OrderView(order_id=order_id, fields=dict(fields), updates=updates)

    def _order_evidence(self, order_id: Optional[str]) -> Tuple[Evidence, ...]:
        if not order_id:
            return ()
        return tuple(self.order_update_evidence.get(order_id, ()))

    # ------------------------------------------------------------------
    # clarifications
    # ------------------------------------------------------------------
    def _add_clarification(
        self,
        turn: Turn,
        request: Optional[_MutableRequest],
        kind: str,
        question: str,
        reason: str,
        candidates: Sequence[str],
        evidence: Sequence[Evidence],
        policy_evidence: Sequence[PolicyEvidence] = (),
        condition_code: Optional[str] = None,
        audience: str = "customer",
    ) -> Optional[Clarification]:
        key = (request.request_id if request else "", kind, condition_code or question)
        if key in self._clarification_keys:
            return None
        self._clarification_keys.add(key)
        clarification = Clarification(
            clarification_id=self._next_id("clarify"),
            matter_id=request.matter_id if request else self._active_matter,
            request_id=request.request_id if request else None,
            kind=kind,
            question=question,
            reason=reason,
            candidates=tuple(candidates),
            evidence=tuple(evidence),
            policy_evidence=tuple(item for item in policy_evidence if item is not None),
            condition_code=condition_code,
            audience=audience,
        )
        self.clarifications.append(clarification)
        if request is not None:
            request.clarification_ids.append(clarification.clarification_id)
            request.progress.append(
                self._progress(
                    turn=turn,
                    kind="clarification_raised",
                    description=f"需追问：{question}",
                    request_id=request.request_id,
                    matter_id=request.matter_id,
                    evidence=clarification.evidence[0] if clarification.evidence else None,
                )
            )
        return clarification

    # ------------------------------------------------------------------
    # finalisation
    # ------------------------------------------------------------------
    def _finalize_matters(self) -> None:
        for matter_id, matter in self.matters.items():
            requests = [item for item in self.requests if item.matter_id == matter_id]
            if not requests and not matter["request_ids"]:
                continue
            statuses = [item.status for item in requests]
            escalations = [
                item for item in self.escalations if item.matter_id == matter_id
            ]
            escalation_states = {item.status for item in escalations}
            if requests and all(status == "completed" for status in statuses):
                status = "completed"
            elif "needs_review" in escalation_states:
                status = "needs_review"
            elif "in_progress" in escalation_states or any(
                state == "escalated" for state in statuses
            ):
                status = "escalated"
            elif escalation_states:
                status = "escalation_recommended"
            elif any(state == "promised" for state in statuses):
                status = "promised"
            elif any(state == "open" for state in statuses):
                status = "open"
            elif requests:
                status = "completed"
            else:
                status = "open"
            matter["status"] = status
            matter["promise_ids"] = unique(
                [promise.promise_id for promise in self.promises if promise.matter_id == matter_id]
            )
            matter["repeated_promise_ids"] = unique(
                [
                    promise.promise_id
                    for promise in self.promises
                    if promise.matter_id == matter_id and promise.status == "repeated"
                ]
            )
            matter["clarification_ids"] = unique(
                [
                    clarification.clarification_id
                    for clarification in self.clarifications
                    if clarification.matter_id == matter_id
                ]
            )
            matter["intents"] = unique([item.intent for item in requests])
            matter["escalation_states"] = [item.status for item in escalations]

    def _build_ledger(self) -> DialogueLedger:
        frozen_requests = [item.freeze() for item in self.requests]
        frozen_promises = [item.freeze() for item in self.promises]
        matters = []
        for matter_id, data in self.matters.items():
            requests = [item for item in frozen_requests if item.matter_id == matter_id]
            if not requests:
                continue
            next_action = _next_action(requests, frozen_promises, self.clarifications, data)
            matters.append(
                Matter(
                    matter_id=matter_id,
                    label=data["label"],
                    order_ids=tuple(data["order_ids"]),
                    intents=tuple(data.get("intents", [])),
                    request_ids=tuple(item.request_id for item in requests),
                    promise_ids=tuple(data.get("promise_ids", [])),
                    repeated_promise_ids=tuple(data.get("repeated_promise_ids", [])),
                    escalation_ids=tuple(data.get("escalation_ids", [])),
                    clarification_ids=tuple(data.get("clarification_ids", [])),
                    status=data.get("status", "open"),
                    status_label=STATUS_LABELS.get(data.get("status", "open"), ""),
                    next_action=next_action,
                    evidence=tuple(
                        evidence
                        for request in requests
                        for evidence in request.evidence
                    ),
                )
            )

        orders = {
            order_id: self._order_view(order_id)
            for order_id in self.orders
            if self.orders[order_id]
        }
        unresolved = sum(1 for item in self.clarifications if item.status == "open")
        if unresolved:
            self.notes.append(f"仍有 {unresolved} 个追问项未澄清，下一班需先补齐资料。")
        return DialogueLedger(
            conversation_id=self.conversation_id,
            as_of=self.as_of,
            turns=tuple(self.turns),
            requests=tuple(frozen_requests),
            promises=tuple(frozen_promises),
            matters=tuple(matters),
            assessments=tuple(self.assessments),
            escalations=tuple(self.escalations),
            clarifications=tuple(self.clarifications),
            corrections=tuple(self.corrections),
            order_updates=tuple(self.order_updates),
            progress=tuple(self.progress),
            orders=orders,
            policies=self.policies,
            notes=tuple(self.notes),
        )


# ----------------------------------------------------------------------
# module helpers
# ----------------------------------------------------------------------
def _likely_continuation(clause: str) -> bool:
    return any(
        marker in clause
        for marker in ("麻烦", "请", "尽快", "还要", "另外", "同时", "也", "再", "帮我")
    )


def _text_overlap(left: str, right: str) -> float:
    """Character-level Jaccard overlap, used only to spot echoed sentences."""
    left_chars = {char for char in left if "\u4e00" <= char <= "\u9fff"}
    right_chars = {char for char in right if "\u4e00" <= char <= "\u9fff"}
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / len(left_chars | right_chars)


def unique_requests(
    requests: Sequence[_MutableRequest],
) -> list[_MutableRequest]:
    seen: set[str] = set()
    ordered: list[_MutableRequest] = []
    for request in requests:
        if request.request_id in seen:
            continue
        seen.add(request.request_id)
        ordered.append(request)
    return ordered


def _clause_looks_like_fact(clause: str) -> bool:
    """True for statement fragments that belong to an ongoing correction."""
    if detect_intents(clause) and not _extract_order_fields(clause):
        return False
    if is_question(clause):
        return False
    return bool(_extract_order_fields(clause))


def _order_from_clause(clause: str) -> Optional[str]:
    order_ids = find_order_ids(clause)
    return order_ids[0] if order_ids else None


def _intent_from_clause(clause: str) -> Optional[str]:
    intents = detect_intents(clause)
    return intents[0][0] if intents else None


def _is_escalation_progress(clause: str) -> bool:
    return any(
        marker in clause
        for marker in (
            "升级", "转交", "转给", "转交至", "已反馈给", "已反馈",
            "处理中", "已完成", "已提交审核", "已提交财务",
        )
    )


def _same_commitment(promise: _MutablePromise, clause: str) -> bool:
    if normalize_text(promise.quote) == normalize_text(clause):
        return True
    old = find_deadline(promise.quote)
    new = find_deadline(clause)
    if old and new and old == new:
        # Same deadline alone is not enough: "今天下班前核对承运信息" and
        # "今天下班前提交退款审核" are different commitments.  Require the
        # wording to overlap substantially.
        return _text_overlap(remove_annotations(promise.quote), clause) >= 0.6
    return False


def _correction_intent(
    clause: str,
    found_annotations: Sequence[str],
    requests: Sequence[_MutableRequest],
) -> Optional[str]:
    if found_annotations:
        for item in found_annotations:
            if item.startswith("退款") or item in _REFUND_WORDS:
                return "refund"
            if item.startswith("物流") or item in _LOGISTICS_WORDS:
                return "logistics"
            if "取消" in item:
                return "cancel"
            if "投诉" in item:
                return "complaint"
    # "提到的退款" / "之前的物流" style references.
    for token, intent in (
        ("退款", "refund"),
        ("赔付", "refund"),
        ("物流", "logistics"),
        ("快递", "logistics"),
        ("取消", "cancel"),
        ("投诉", "complaint"),
    ):
        if token in clause:
            return intent
    # Only a generic order correction: look at requests on the named order and
    # keep their intents unchanged (return None).
    return None


_ORDER_FIELD_CUES: Tuple[Tuple[str, str], ...] = (
    ("重复扣款", "duplicate_charge"),
    ("扣款", "charge_status"),
    ("退款", "refund_status"),
    ("退回", "refund_status"),
    ("已到账", "refund_received"),
    ("到账", "refund_received"),
    ("签收", "delivery_status"),
    ("未签收", "delivery_status"),
    ("已签收", "delivery_status"),
    ("承运", "carrier"),
    ("物流", "tracking_status"),
    ("金额", "amount"),
    ("价保", "price_protection"),
)


def _extract_order_fields(clause: str) -> dict[str, Any]:
    """Extract only facts explicitly asserted in a correction clause.

    This is intentionally narrow.  Anything not asserted stays unknown so that
    the policy check asks rather than guesses.
    """
    fields: dict[str, Any] = {}
    text = clause
    if any(phrase in text for phrase in ("不是重复扣款", "非重复扣款", "单次扣款")):
        fields["charge_status"] = "normal"
        fields["refund_proof"] = False
    elif "重复扣款" in text:
        fields["charge_status"] = "duplicate"
    elif "扣款" in text and any(word in text for word in ("正常", "一次")):
        fields["charge_status"] = "normal"
        fields["refund_proof"] = False
    if any(phrase in text for phrase in ("凭证已提供", "已提供凭证", "提供扣款凭证", "凭证已上传")):
        fields["refund_proof"] = True
    if "未签收" in text:
        fields["delivery_status"] = "not_delivered"
    elif "已签收" in text:
        fields["delivery_status"] = "delivered"
    if "已退回" in text or "退款完成" in text or "已退款" in text:
        fields["refund_status"] = "refunded"
        fields["refund_received"] = True
    if "未退款" in text or "没有退款" in text:
        fields["refund_status"] = "not_refunded"
        fields["refund_received"] = False
    if "承运" in text:
        match = None
        for carrier in ("顺丰", "中通", "圆通", "韵达", "京东物流", "EMS"):
            if carrier in text:
                match = carrier
                break
        if match:
            fields["carrier"] = match
    if "金额" in text:
        amount = _extract_amount(text)
        if amount is not None:
            fields["amount"] = amount
    if "发货后" in text or "超过" in text:
        days = _extract_days(text)
        if days is not None:
            fields["days_since_dispatch"] = days
    if "价保" in text:
        if any(word in text for word in ("有", "含", "支持")):
            fields["price_protection"] = True
        if any(word in text for word in ("没有", "无", "不含", "不支持")):
            fields["price_protection"] = False
    if "订单状态" in text and any(word in text for word in ("已取消", "取消")):
        fields["order_status"] = "cancelled"
    return fields


def _extract_amount(text: str) -> Optional[float]:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*元", text)
    if match:
        return float(match.group(1))
    return None


def _extract_days(text: str) -> Optional[int]:
    import re

    match = re.search(r"(\d+)\s*天", text)
    if match:
        return int(match.group(1))
    return None


def _next_action(
    requests: Sequence[Request],
    promises: Sequence[Promise],
    clarifications: Sequence[Clarification],
    matter: Mapping[str, Any],
) -> str:
    matter_id = None
    if requests:
        matter_id = requests[0].matter_id
    open_clarifications = [
        item
        for item in clarifications
        if item.matter_id == matter_id and item.status == "open"
    ]
    if requests and all(item.status == "completed" for item in requests):
        return "事项已完成，保留原话与政策依据；如仍有升级工单，请确认关闭。"
    if open_clarifications:
        customer = [i.question for i in open_clarifications if i.audience == "customer"]
        internal = [i.question for i in open_clarifications if i.audience != "customer"]
        pieces = []
        if customer:
            pieces.append("先向客户追问：" + "；".join(customer[:2]))
        if internal:
            pieces.append("内部先核对：" + "；".join(internal[:2]))
        return "；".join(pieces)
    escalation_states = set(matter.get("escalation_states", ()))
    if "needs_review" in escalation_states:
        return "升级工单需人工复核政策条件；已确认承诺保留原文和确认时间，由复核人决定后续。"
    if "in_progress" in escalation_states:
        if any(item.status == "promised" for item in requests):
            return "跟进升级团队进度，同时按已确认承诺原文履行，不要重复承诺时限。"
        return "跟进升级团队的处理进度，不要重复承诺时限。"
    if "recommended" in escalation_states:
        if any(item.status == "promised" for item in requests):
            return "确认是否转交升级团队；同时按已确认承诺的时间履行，不要重复承诺。"
        return "确认是否转交对应团队升级，不要仅凭政策允许就当作已批准。"
    if any(item.status == "promised" for item in requests):
        return "按已确认承诺的原文和确认时间履行，不要改写下班次的承诺。"
    if any(item.status == "open" for item in requests):
        return "按待处理诉求继续处理；涉及政策条件时先核对订单资料。"
    return "无需进一步处理，保留原话与政策依据备查。"


def analyse_file(
    dialogue_path,
    policy_path=None,
    orders_path=None,
    as_of: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> DialogueLedger:
    """Convenience entry point used by the CLI and by tests."""
    import json
    from pathlib import Path as _Path

    turns = load_turns(dialogue_path)
    policies = load_policies(policy_path) if policy_path else ()
    orders: dict[str, dict[str, Any]] = {}
    if orders_path:
        raw = json.loads(_Path(orders_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("orders file must be a JSON object keyed by order id")
        for order_id, fields in raw.items():
            if not isinstance(fields, dict):
                raise ValueError(f"order {order_id!r} must be an object")
            orders[str(order_id)] = dict(fields)
    identifier = conversation_id or _Path(dialogue_path).stem
    return ConversationAnalyser(
        turns=turns,
        policies=policies,
        orders=orders,
        as_of=as_of,
        conversation_id=identifier,
    ).run()
