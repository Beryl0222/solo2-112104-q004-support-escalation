"""Domain objects for the read-only conversation ledger.

The objects in this module deliberately keep quotes and their offsets next to
every derived fact.  A support lead should be able to jump from a request, a
promise, a policy conclusion or an escalation back to the exact sentence that
produced it.

Nothing in this package sends messages.  ``DialogueLedger`` is an audit view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Turn:
    """One original utterance, exactly as supplied by the source record."""

    record_id: str
    turn: int
    channel: str
    speaker: str
    text: str
    timestamp: Optional[str] = None
    speaker_label: Optional[str] = None
    kind: str = "utterance"


@dataclass(frozen=True)
class Evidence:
    """A character-exact quote inside one dialogue turn."""

    record_id: str
    turn: int
    channel: str
    speaker: str
    quote: str
    start: int
    end: int
    timestamp: Optional[str] = None
    speaker_label: Optional[str] = None


@dataclass(frozen=True)
class PolicyEvidence:
    """A quote from a policy document (not invented from the dialogue)."""

    policy_id: str
    version: str
    source: str
    quote: str
    condition_code: Optional[str] = None


@dataclass(frozen=True)
class PolicyCondition:
    code: str
    description: str
    field: str
    operator: str
    value: Any
    prompt: str
    source_quote: str
    audience: str = "customer"  # customer / internal


@dataclass(frozen=True)
class Policy:
    policy_id: str
    version: str
    applies_to: Tuple[str, ...]
    allows: bool
    action: str
    summary: str
    source: str
    conditions: Tuple[PolicyCondition, ...] = ()
    effective_at: Optional[str] = None


@dataclass(frozen=True)
class PolicyConditionResult:
    condition: PolicyCondition
    status: str  # met / unmet / unknown
    actual: Any
    detail: str
    source_evidence: Optional[PolicyEvidence] = None


@dataclass(frozen=True)
class PolicyAssessment:
    """One immutable policy check; earlier checks are never overwritten."""

    assessment_id: str
    request_id: str
    matter_id: str
    policy_id: Optional[str]
    policy_version: Optional[str]
    policy_allows: Optional[bool]
    eligible: str  # yes / no / unknown
    conclusion: str
    conditions: Tuple[PolicyConditionResult, ...]
    missing_conditions: Tuple[PolicyConditionResult, ...]
    checked_turn: int
    checked_at: Optional[str]
    trigger: str
    trigger_evidence: Optional[Evidence]
    policy_evidence: Tuple[PolicyEvidence, ...]
    order_id: Optional[str]
    order_fields: Mapping[str, Any]
    order_evidence: Tuple[Evidence, ...] = ()
    policy_summary: str = ""


@dataclass(frozen=True)
class OrderFieldUpdate:
    update_id: str
    order_id: str
    fields: Mapping[str, Any]
    turn: int
    evidence: Evidence


@dataclass(frozen=True)
class OrderView:
    order_id: str
    fields: Mapping[str, Any]
    updates: Tuple[OrderFieldUpdate, ...] = ()


@dataclass(frozen=True)
class OrderAssociation:
    order_id: Optional[str]
    method: str  # explicit / anaphora / correction / unknown
    evidence: Optional[Evidence]
    note: str = ""


@dataclass(frozen=True)
class Clarification:
    clarification_id: str
    matter_id: Optional[str]
    request_id: Optional[str]
    kind: str
    question: str
    reason: str
    candidates: Tuple[str, ...]
    evidence: Tuple[Evidence, ...]
    policy_evidence: Tuple[PolicyEvidence, ...] = ()
    condition_code: Optional[str] = None
    audience: str = "customer"  # customer / internal
    status: str = "open"  # open / resolved
    resolution: Optional[str] = None


@dataclass(frozen=True)
class ProgressEvent:
    event_id: str
    request_id: Optional[str]
    matter_id: Optional[str]
    kind: str
    description: str
    turn: int
    timestamp: Optional[str]
    evidence: Optional[Evidence]


@dataclass(frozen=True)
class Request:
    request_id: str
    matter_id: str
    intent: str
    intent_label: str
    order_id: Optional[str]
    evidence: Tuple[Evidence, ...]
    correction_evidence: Tuple[Evidence, ...]
    associations: Tuple[OrderAssociation, ...]
    status: str
    status_label: str
    status_evidence: Optional[Evidence]
    dissatisfaction: bool
    first_turn: int
    last_turn: int
    policy_assessments: Tuple[PolicyAssessment, ...]
    active_assessment_id: Optional[str]
    clarification_ids: Tuple[str, ...]
    escalation_id: Optional[str]
    progress: Tuple[ProgressEvent, ...]


@dataclass(frozen=True)
class Promise:
    """A frozen agent commitment.

    ``confirmed_at`` and ``quote`` are snapshots.  A later policy revision can
    add a new assessment, but it never edits this object.
    """

    promise_id: str
    matter_id: Optional[str]
    request_id: Optional[str]
    intent: Optional[str]
    intent_label: Optional[str]
    order_id: Optional[str]
    quote: str
    evidence: Evidence
    confirmed_turn: int
    confirmed_at: Optional[str]
    deadline_text: Optional[str]
    policy_version_at_confirmation: Optional[str]
    duplicate_of_promise_id: Optional[str] = None
    superseded_by_promise_id: Optional[str] = None
    status: str = "active"  # active / repeated / restated / adjusted


@dataclass(frozen=True)
class Correction:
    correction_id: str
    kind: str  # binding / restatement / order_update
    turn: int
    intent: Optional[str]
    order_id: Optional[str]
    fields: Mapping[str, Any]
    affected_request_ids: Tuple[str, ...]
    evidence: Evidence
    note: str


@dataclass(frozen=True)
class Escalation:
    escalation_id: str
    matter_id: str
    request_id: str
    team: str
    team_label: str
    reason: str
    status: str  # recommended / in_progress / completed
    created_turn: int
    updated_turn: int
    evidence: Tuple[Evidence, ...]
    review_note: Optional[str] = None


@dataclass(frozen=True)
class Matter:
    matter_id: str
    label: str
    order_ids: Tuple[str, ...]
    intents: Tuple[str, ...]
    request_ids: Tuple[str, ...]
    promise_ids: Tuple[str, ...]
    repeated_promise_ids: Tuple[str, ...]
    escalation_ids: Tuple[str, ...]
    clarification_ids: Tuple[str, ...]
    status: str
    status_label: str
    next_action: str
    evidence: Tuple[Evidence, ...]


@dataclass(frozen=True)
class DialogueLedger:
    conversation_id: str
    as_of: Optional[str]
    turns: Tuple[Turn, ...]
    requests: Tuple[Request, ...]
    promises: Tuple[Promise, ...]
    matters: Tuple[Matter, ...]
    assessments: Tuple[PolicyAssessment, ...]
    escalations: Tuple[Escalation, ...]
    clarifications: Tuple[Clarification, ...]
    corrections: Tuple[Correction, ...]
    order_updates: Tuple[OrderFieldUpdate, ...]
    progress: Tuple[ProgressEvent, ...]
    orders: Mapping[str, OrderView]
    policies: Tuple[Policy, ...]
    notes: Tuple[str, ...] = ()

    # -- lookup helpers -------------------------------------------------
    def matter(self, matter_id: str) -> Matter:
        return next(item for item in self.matters if item.matter_id == matter_id)

    def request(self, request_id: str) -> Request:
        return next(item for item in self.requests if item.request_id == request_id)

    def promise(self, promise_id: str) -> Promise:
        return next(item for item in self.promises if item.promise_id == promise_id)

    def assessment(self, assessment_id: str) -> PolicyAssessment:
        return next(item for item in self.assessments if item.assessment_id == assessment_id)

    def turn(self, record_id: str) -> Turn:
        return next(item for item in self.turns if item.record_id == record_id)

    def locate(self, evidence: Evidence) -> str:
        """Stable pointer back to the source sentence.

        The ``record#turn`` part is human-readable; the offsets allow a UI to
        highlight the exact characters without re-running the analyser.
        """
        return (
            f"{evidence.record_id}#turn-{evidence.turn}"
            f"@{evidence.start}:{evidence.end}"
        )

    def open_clarifications(self) -> Tuple[Clarification, ...]:
        return tuple(item for item in self.clarifications if item.status == "open")

    def active_assessment(self, request: Request) -> Optional[PolicyAssessment]:
        if request.active_assessment_id is None:
            return None
        return self.assessment(request.active_assessment_id)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def render_text(self) -> str:
        from .report import render_text

        return render_text(self)

    def summary(self) -> dict[str, object]:
        """Small hand-over summary for dashboards/tests."""
        return {
            "conversation_id": self.conversation_id,
            "turns": len(self.turns),
            "matters": len(self.matters),
            "open_requests": sum(1 for item in self.requests if item.status == "open"),
            "promised": sum(1 for item in self.promises if item.status == "active"),
            "repeated_promises": sum(1 for item in self.promises if item.status == "repeated"),
            "open_clarifications": len(self.open_clarifications()),
            "escalations": len(self.escalations),
            "needs_review": sum(1 for item in self.escalations if item.status == "needs_review"),
        }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
