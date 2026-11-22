"""Synthetic policy catalogue and condition checks.

A policy says what an action would require.  It never says that a particular
order met those requirements: that is decided from the order snapshot and its
later corrections, one condition at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .models import (
    OrderView,
    Policy,
    PolicyCondition,
    PolicyConditionResult,
    PolicyEvidence,
)


class PolicyDataError(ValueError):
    """Raised when the synthetic policy catalogue is malformed."""


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: Policy
    results: Tuple[PolicyConditionResult, ...]
    eligible: str  # yes / no / unknown
    missing: Tuple[PolicyConditionResult, ...]
    failed: Tuple[PolicyConditionResult, ...]

    @property
    def conclusion(self) -> str:
        allows = "允许" if self.policy.allows else "不允许"
        if self.eligible == "yes":
            return (
                f"政策 {self.policy.policy_id}@{self.policy.version} {allows}"
                f"{self.policy.action}，且订单信息满足全部必要条件。"
            )
        if self.eligible == "no":
            failed = "、".join(result.condition.description for result in self.failed)
            return (
                f"政策 {self.policy.policy_id}@{self.policy.version} {allows}"
                f"{self.policy.action}，但订单不满足条件：{failed}。"
                "政策允许不等于本订单符合条件。"
            )
        missing = "、".join(result.condition.description for result in self.missing)
        return (
            f"政策 {self.policy.policy_id}@{self.policy.version} {allows}"
            f"{self.policy.action}，但缺少订单信息：{missing}；暂不能判定是否符合条件。"
        )


def _require(data: Mapping[str, Any], key: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise PolicyDataError(f"missing field {key!r} in policy document") from exc


def load_policies(path: Path | str) -> Tuple[Policy, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise PolicyDataError("policy catalogue must be a JSON list")
    policies: list[Policy] = []
    for document in raw:
        conditions: list[PolicyCondition] = []
        for item in document.get("conditions", []):
            conditions.append(
                PolicyCondition(
                    code=_require(item, "code"),
                    description=_require(item, "description"),
                    field=_require(item, "field"),
                    operator=_require(item, "operator"),
                    value=item.get("value"),
                    prompt=_require(item, "prompt"),
                    source_quote=_require(item, "source_quote"),
                    audience=str(item.get("audience", "customer")),
                )
            )
        policies.append(
            Policy(
                policy_id=_require(document, "policy_id"),
                version=_require(document, "version"),
                applies_to=tuple(document.get("applies_to", [])),
                allows=bool(document.get("allows", False)),
                action=document.get("action", ""),
                summary=_require(document, "summary"),
                source=_require(document, "source"),
                conditions=tuple(conditions),
                effective_at=document.get("effective_at"),
            )
        )
    return tuple(policies)


def select_policy(
    policies: Sequence[Policy], intent: str, as_of: Optional[str]
) -> Optional[Policy]:
    """Choose the policy version in force for ``intent`` at ``as_of``.

    Versions are ordered by ``effective_at`` (ISO strings compare correctly).
    When the assessment has no clock (``as_of is None``) the original version
    is used, which keeps offline replay deterministic.
    """
    candidates = [policy for policy in policies if intent in policy.applies_to]
    if not candidates:
        return None
    if as_of is None:
        return min(candidates, key=lambda item: item.effective_at or "")
    eligible = [
        policy
        for policy in candidates
        if policy.effective_at is None or policy.effective_at <= as_of
    ]
    if not eligible:
        return min(candidates, key=lambda item: item.effective_at or "")
    return max(eligible, key=lambda item: item.effective_at or "")


def policy_history(policies: Sequence[Policy], intent: str) -> Tuple[Policy, ...]:
    return tuple(
        sorted(
            (policy for policy in policies if intent in policy.applies_to),
            key=lambda item: item.effective_at or "",
        )
    )


def evaluate_order(
    policy: Policy,
    order: Optional[OrderView],
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> PolicyEvaluation:
    fields = dict(order.fields) if order else {}
    if extra_fields:
        fields.update(extra_fields)
    results: list[PolicyConditionResult] = []
    for condition in policy.conditions:
        if condition.field not in fields:
            actual: Any = None
            status = "unknown"
        else:
            actual = fields[condition.field]
            status = "met" if _condition_holds(condition, actual) else "unmet"
        detail = _describe(condition, status, actual)
        results.append(
            PolicyConditionResult(
                condition=condition,
                status=status,
                actual=actual,
                detail=detail,
                source_evidence=PolicyEvidence(
                    policy_id=policy.policy_id,
                    version=policy.version,
                    source=policy.source,
                    quote=condition.source_quote,
                    condition_code=condition.code,
                ),
            )
        )

    missing = tuple(result for result in results if result.status == "unknown")
    failed = tuple(result for result in results if result.status == "unmet")
    if failed:
        eligible = "no"
    elif missing:
        eligible = "unknown"
    else:
        eligible = "yes"
    return PolicyEvaluation(
        policy=policy,
        results=tuple(results),
        eligible=eligible,
        missing=missing,
        failed=failed,
    )


def _condition_holds(condition: PolicyCondition, actual: Any) -> bool:
    operator = condition.operator
    expected = condition.value
    if operator == "truthy":
        return bool(actual)
    if operator == "falsy":
        return not bool(actual)
    if operator == "equals":
        return actual == expected
    if operator == "gte":
        try:
            return float(actual) >= float(expected)
        except (TypeError, ValueError):
            return False
    if operator == "lte":
        try:
            return float(actual) <= float(expected)
        except (TypeError, ValueError):
            return False
    if operator == "not_equals":
        return actual != expected
    if operator == "contains":
        return str(expected) in str(actual)
    raise PolicyDataError(f"unknown condition operator {operator!r}")


def _describe(condition: PolicyCondition, status: str, actual: Any) -> str:
    if status == "unknown":
        return f"{condition.description}：资料缺失（需要：{condition.prompt}）"
    verdict = "满足" if status == "met" else "不满足"
    return f"{condition.description}：{verdict}（当前值：{actual!r}）"
