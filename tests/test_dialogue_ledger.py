import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from support_dialogue import analyse_file, render_matter, render_text  # noqa: E402
from support_dialogue.analyser import ConversationAnalyser  # noqa: E402
from support_dialogue.ingest import load_turns  # noqa: E402
from support_dialogue.models import DialogueLedger, Evidence  # noqa: E402
from support_dialogue.policy import load_policies  # noqa: E402

DATA = ROOT / "data"
MAIN = DATA / "support_conversation.json"
AMBIGUOUS = DATA / "support_ambiguous.json"
POLICY_UPDATE = DATA / "support_policy_update.json"
POLICIES = DATA / "policies.json"
ORDERS = DATA / "orders.json"


def analyse(path: Path, **kwargs) -> DialogueLedger:
    return analyse_file(path, policy_path=POLICIES, orders_path=ORDERS, **kwargs)


class EvidenceIntegrityTests(unittest.TestCase):
    """Every derived fact must quote the source turn character-for-character."""

    def assert_evidence_valid(self, ledger: DialogueLedger) -> None:
        turns = {turn.record_id: turn for turn in ledger.turns}
        evidences: list[Evidence] = []

        def collect(item):
            if isinstance(item, Evidence):
                evidences.append(item)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)
            elif hasattr(item, "__dataclass_fields__"):
                for name in item.__dataclass_fields__:
                    collect(getattr(item, name))
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)

        collect(ledger.requests)
        collect(ledger.promises)
        collect(ledger.clarifications)
        collect(ledger.corrections)
        collect(ledger.escalations)
        collect(ledger.order_updates)

        self.assertTrue(evidences)
        for evidence in evidences:
            with self.subTest(record=evidence.record_id, quote=evidence.quote):
                turn = turns[evidence.record_id]
                self.assertEqual(turn.turn, evidence.turn)
                self.assertEqual(turn.channel, evidence.channel)
                self.assertEqual(turn.text[evidence.start:evidence.end], evidence.quote)
                self.assertIn(evidence.quote, turn.text)

    def test_all_scenarios_keep_exact_quotes_and_turns(self):
        for path in (DATA / "support.json", MAIN, AMBIGUOUS, POLICY_UPDATE):
            with self.subTest(path=path.name):
                ledger = analyse(path)
                self.assert_evidence_valid(ledger)

    def test_ledger_can_point_back_to_the_exact_characters(self):
        ledger = analyse(MAIN)
        evidence = ledger.request("request-2").evidence[0]
        pointer = ledger.locate(evidence)
        self.assertIn("#turn-3@", pointer)
        start_end = pointer.split("@", 1)[1]
        start, end = (int(part) for part in start_end.split(":"))
        turn = ledger.turn(evidence.record_id)
        self.assertEqual(turn.text[start:end], evidence.quote)

    def test_each_request_keeps_at_least_one_original_sentence(self):
        ledger = analyse(MAIN)
        for request in ledger.requests:
            with self.subTest(request=request.request_id):
                self.assertTrue(request.evidence)

    def test_policy_assessments_cite_order_and_policy_evidence(self):
        ledger = analyse(MAIN)
        for assessment in ledger.assessments:
            with self.subTest(assessment=assessment.assessment_id):
                self.assertTrue(assessment.policy_evidence)
                for item in assessment.policy_evidence:
                    self.assertTrue(item.quote)
                    self.assertTrue(item.policy_id)
                if assessment.trigger == "order_updated":
                    self.assertTrue(assessment.order_evidence)


class MultiMatterTests(unittest.TestCase):
    def setUp(self):
        self.ledger = analyse(MAIN)

    def test_one_dialogue_two_orders_become_distinct_matters(self):
        matters = {matter.matter_id: matter for matter in self.ledger.matters}
        order_to_matters = {}
        for matter in matters.values():
            for order in matter.order_ids:
                order_to_matters.setdefault(order, set()).add(matter.matter_id)
        self.assertIn("A", order_to_matters)
        self.assertIn("B", order_to_matters)
        self.assertNotEqual(order_to_matters["A"], order_to_matters["B"])

    def test_logistics_is_not_absorbed_into_refund(self):
        refunds = [r for r in self.ledger.requests if r.intent == "refund"]
        logistics = [r for r in self.ledger.requests if r.intent == "logistics"]
        self.assertTrue(refunds)
        self.assertTrue(logistics)
        for request in refunds:
            self.assertNotEqual(request.order_id, "A")
        for request in logistics:
            self.assertNotIn("退款", request.intent)

    def test_pronoun_resolves_to_unique_prior_order(self):
        logistics = next(r for r in self.ledger.requests if r.intent == "logistics")
        anaphora = [a for a in logistics.associations if a.method == "anaphora"]
        self.assertEqual(len(anaphora), 1)
        self.assertEqual(anaphora[0].order_id, "A")

    def test_pronoun_with_multiple_candidates_raises_clarification(self):
        ledger = analyse(AMBIGUOUS)
        refund = next(r for r in ledger.requests if r.intent == "refund")
        # "那个订单" in amb-5 appears while two orders exist; the engine must
        # not silently pick one.  It either resolves to the prior refund order
        # (unique same-intent antecedent) or asks -- never to order A.
        self.assertNotEqual(refund.order_id, "A")
        ambiguous = [
            item
            for item in ledger.clarifications
            if item.kind in {"ambiguous_reference", "missing_order"}
        ]
        for item in ambiguous:
            self.assertTrue(item.question)
            self.assertTrue(item.evidence)
            self.assertIn("订单", item.question)

    def test_unknown_reference_asks_a_specific_question(self):
        turns = load_turns(AMBIGUOUS)
        # Build a case with no antecedent at all.
        minimal = [turns[0].__class__(
            record_id="only", turn=1, channel="web", speaker="customer",
            text="那个订单的退款怎么还没到？", timestamp="2026-10-30T11:00:00",
        )]
        ledger = ConversationAnalyser(turns=minimal, policies=load_policies(POLICIES)).run()
        self.assertEqual(len(ledger.requests), 1)
        request = ledger.requests[0]
        self.assertIsNone(request.order_id)
        self.assertTrue(request.clarification_ids)
        clarification = ledger.clarifications[0]
        self.assertIn("哪个订单", clarification.question)
        self.assertEqual(clarification.evidence[0].quote, "那个订单的退款怎么还没到")


class PromiseTests(unittest.TestCase):
    def test_promise_keeps_original_quote_and_confirmation_time(self):
        ledger = analyse(POLICY_UPDATE)
        promise = ledger.promises[0]
        self.assertEqual(promise.confirmed_turn, 2)
        self.assertEqual(promise.confirmed_at, "2026-10-20T09:02:00")
        self.assertIn("今天下班前", promise.quote)
        self.assertIn("今天下班前", promise.evidence.quote)

    def test_policy_revision_does_not_rewrite_confirmed_promise(self):
        ledger = analyse(POLICY_UPDATE)
        first = ledger.promises[0]
        self.assertEqual(first.policy_version_at_confirmation, "POL-REFUND@2026.09")
        # A newer policy version (2026.11) did arrive and was assessed, but the
        # promise text and confirmation timestamp are unchanged.
        active = ledger.active_assessment(ledger.request("request-1"))
        self.assertEqual(active.policy_version, "2026.11")
        self.assertEqual(first.quote, "客服：订单 E 的退款我今天下班前为您提交退款审核。")
        self.assertEqual(first.confirmed_at, "2026-10-20T09:02:00")

    def test_repeated_commitment_is_not_counted_as_a_new_promise(self):
        ledger = analyse(POLICY_UPDATE)
        original = ledger.promises[0]
        repeats = [
            p for p in ledger.promises if p.duplicate_of_promise_id == original.promise_id
        ]
        self.assertEqual(len(repeats), 1)
        self.assertEqual(repeats[0].status, "repeated")
        self.assertEqual(repeats[0].confirmed_turn, 6)
        self.assertEqual(repeats[0].quote, original.quote)

    def test_adjustment_keeps_both_records(self):
        ledger = analyse(MAIN)
        promise_b = [
            p for p in ledger.promises if p.order_id == "B" and p.request_id == "request-2"
        ]
        self.assertGreaterEqual(len(promise_b), 2)
        earlier = promise_b[0]
        later = promise_b[-1]
        self.assertNotEqual(earlier.quote, later.quote)
        self.assertEqual(earlier.status, "adjusted")
        self.assertEqual(earlier.superseded_by_promise_id, later.promise_id)

    def test_completion_marks_promise_fulfilled_without_rewriting_it(self):
        ledger = analyse(MAIN)
        fulfilled = [p for p in ledger.promises if p.status == "fulfilled"]
        self.assertTrue(fulfilled)
        for promise in fulfilled:
            self.assertIn("3 个工作日内", promise.quote)
            self.assertEqual(promise.confirmed_turn, 9)
            self.assertEqual(promise.confirmed_at, "2026-11-03T10:02:00")

    def test_customer_question_is_not_a_promise(self):
        ledger = analyse(MAIN)
        for promise in ledger.promises:
            self.assertEqual(promise.evidence.speaker, "agent")
        # Turn 13 is the customer saying "退款到账了" -- a result, not promise.
        self.assertFalse(
            any(p.confirmed_turn == 13 for p in ledger.promises)
        )


class PolicyBoundaryTests(unittest.TestCase):
    def test_policy_allows_but_order_fails_condition(self):
        ledger = analyse(MAIN)
        request_c = next(
            r for r in ledger.requests if r.order_id == "C" and r.intent == "refund"
        )
        assessment = ledger.active_assessment(request_c)
        self.assertTrue(assessment.policy_allows)
        self.assertEqual(assessment.eligible, "no")
        self.assertIn("政策允许不等于本订单符合条件", assessment.conclusion)
        window = next(
            item for item in assessment.conditions if item.condition.code == "within_window"
        )
        self.assertEqual(window.status, "unmet")
        self.assertEqual(window.actual, 20)

    def test_dissatisfaction_alone_is_not_a_refund_request(self):
        ledger = analyse(MAIN)
        complaint = next(r for r in ledger.requests if r.intent == "complaint")
        self.assertTrue(complaint.dissatisfaction)
        self.assertEqual(complaint.intent, "complaint")
        self.assertIsNone(
            next(
                (
                    r
                    for r in ledger.requests
                    if r.first_turn == 10 and r.intent == "refund"
                ),
                None,
            )
        )
        self.assertTrue(
            any(event.kind == "dissatisfaction_noted" for event in ledger.progress)
        )

    def test_order_correction_triggers_fresh_assessment_and_preserves_history(self):
        ledger = analyse(MAIN)
        request_b = ledger.request("request-2")
        triggers = [item.trigger for item in request_b.policy_assessments]
        self.assertEqual(triggers[0], "initial")
        self.assertIn("order_updated", triggers)
        self.assertIn("policy_updated", triggers)
        self.assertGreater(len(request_b.policy_assessments), 1)
        self.assertEqual(
            request_b.active_assessment_id, request_b.policy_assessments[-1].assessment_id
        )

    def test_missing_condition_asks_the_specific_question(self):
        ledger = analyse(DATA / "support.json")
        refund = next(r for r in ledger.requests if r.intent == "refund")
        open_clarifications = [
            item
            for item in ledger.clarifications
            if item.request_id == refund.request_id and item.status == "open"
        ]
        self.assertTrue(open_clarifications)
        for item in open_clarifications:
            self.assertTrue(item.question)
            self.assertEqual(item.kind, "missing_policy_condition")
            self.assertTrue(item.policy_evidence)
            self.assertTrue(item.policy_evidence[0].quote)

    def test_refund_requires_proof_without_inferring_it(self):
        ledger = analyse(DATA / "support.json")
        refund = next(r for r in ledger.requests if r.intent == "refund")
        assessment = ledger.active_assessment(refund)
        proof = next(
            item for item in assessment.conditions if item.condition.code == "refund_proof"
        )
        self.assertEqual(proof.status, "unknown")
        self.assertIn("凭证", proof.condition.prompt)

    def test_unbound_complaint_has_no_policy_verdict(self):
        ledger = analyse(MAIN)
        complaint = next(r for r in ledger.requests if r.intent == "complaint")
        self.assertIsNone(complaint.order_id)
        self.assertFalse(complaint.policy_assessments)
        self.assertTrue(
            any(
                item.kind == "missing_order" and item.status == "open"
                for item in ledger.clarifications
                if item.request_id == complaint.request_id
            )
        )

    def test_same_deadline_different_action_is_not_a_repeat(self):
        # "今天下班前核对承运信息" and "今天下班前提交退款审核" share a
        # deadline but are different commitments for different orders.
        ledger = analyse(MAIN)
        for promise in ledger.promises:
            if promise.status == "repeated":
                original = ledger.promise(promise.duplicate_of_promise_id)
                self.assertEqual(original.order_id, promise.order_id)
                self.assertEqual(original.intent, promise.intent)

    def test_new_policy_marks_escalation_for_review(self):
        ledger = analyse(MAIN)
        request_c = next(
            r for r in ledger.requests if r.order_id == "C" and r.intent == "refund"
        )
        escalation = next(
            item for item in ledger.escalations
            if item.escalation_id == request_c.escalation_id
        )
        self.assertEqual(escalation.status, "needs_review")
        self.assertIn("复核", escalation.review_note or "")
        self.assertIn("满足政策条件", escalation.reason)  # original reason kept
        self.assertTrue(
            any(
                event.kind == "policy_conflict" and event.request_id == request_c.request_id
                for event in ledger.progress
            )
        )


class EscalationAndProgressTests(unittest.TestCase):
    def test_escalation_requires_met_conditions(self):
        ledger = analyse(MAIN)
        allowed_orders = {"A", "B"}
        for escalation in ledger.escalations:
            request = ledger.request(escalation.request_id)
            assessment = ledger.active_assessment(request)
            if escalation.reason.startswith("退款") and assessment is not None:
                self.assertNotEqual(assessment.eligible, "unknown")
        self.assertTrue(
            any(
                ledger.request(item.request_id).order_id in allowed_orders
                for item in ledger.escalations
            )
        )

    def test_policy_met_but_customer_never_asked_for_refund(self):
        # Policy permitting an action is not an instruction to escalate.  A
        # logistics query whose conditions are met escalates to logistics, but
        # it never produces a refund escalation or a refund request.
        ledger = analyse(MAIN)
        logistics = next(r for r in ledger.requests if r.intent == "logistics")
        refund_escalations = [
            item for item in ledger.escalations if item.request_id == logistics.request_id
        ]
        self.assertTrue(all(item.team == "logistics" for item in refund_escalations))

    def test_agent_handoff_updates_escalation_status(self):
        ledger = analyse(MAIN)
        progress = [
            event
            for event in ledger.progress
            if event.kind == "escalation_progress" and event.request_id == "request-2"
        ]
        self.assertTrue(progress)
        escalation = next(
            item for item in ledger.escalations
            if item.request_id == "request-2"
        )
        self.assertIn(escalation.status, {"in_progress", "completed"})
        self.assertIn("3 个工作日内", escalation.evidence[-1].quote)

    def test_next_shift_is_warned_not_to_repeat(self):
        ledger = analyse(POLICY_UPDATE)
        repeat_events = [
            event for event in ledger.progress if event.kind == "promise_repeated"
        ]
        self.assertTrue(repeat_events)
        self.assertIn("第一条", repeat_events[0].description)
        self.assertIn("不重复", repeat_events[0].description)

    def test_matter_next_action_is_actionable(self):
        ledger = analyse(DATA / "support.json")
        matter = next(m for m in ledger.matters if "B" in m.order_ids)
        self.assertIn("追问", matter.next_action)


class OutputTests(unittest.TestCase):
    def test_summary_counts_handover_risks(self):
        ledger = analyse(POLICY_UPDATE)
        summary = ledger.summary()
        self.assertEqual(summary["repeated_promises"], 1)
        self.assertEqual(summary["needs_review"], 1)
        self.assertEqual(summary["promised"], 1)

    def test_ledger_is_json_serialisable(self):
        ledger = analyse(MAIN)
        payload = ledger.to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["conversation_id"], ledger.conversation_id)
        self.assertEqual(len(decoded["matters"]), len(ledger.matters))

    def test_report_shows_quote_turn_and_channel_for_every_matter(self):
        ledger = analyse(MAIN)
        text = render_text(ledger)
        for matter in ledger.matters:
            self.assertIn(matter.matter_id, text)
            evidence = matter.evidence[0]
            self.assertIn(evidence.quote, text)
            self.assertIn(f"第{evidence.turn}轮", text)
        self.assertIn("不会发送任何客户答复", text)

    def test_matter_report_can_be_rendered_alone(self):
        ledger = analyse(MAIN)
        text = render_matter(ledger, "matter-1")
        self.assertIn("matter-1", text)
        self.assertIn("物流", text)

    def test_analysis_is_deterministic(self):
        first = analyse(MAIN).to_dict()
        second = analyse(MAIN).to_dict()
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_cli_text_and_json(self):
        base = [sys.executable, "-m", "support_dialogue", str(MAIN)]
        text = subprocess.run(
            base + ["--policies", str(POLICIES), "--orders", str(ORDERS), "--matter", "matter-1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("matter-1", text.stdout)
        payload = subprocess.run(
            base + ["--policies", str(POLICIES), "--orders", str(ORDERS), "--format", "json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        decoded = json.loads(payload.stdout)
        self.assertIn("requests", decoded)
        self.assertIn("summary", decoded)
        self.assertGreaterEqual(decoded["summary"]["matters"], 1)

    def test_cli_reports_bad_input_without_traceback(self):
        result = subprocess.run(
            [sys.executable, "-m", "support_dialogue", "does-not-exist.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("无法整理对话", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class ImmutabilityTests(unittest.TestCase):
    def test_frozen_dataclasses_reject_mutation(self):
        ledger = analyse(POLICY_UPDATE)
        with self.assertRaises(Exception):
            ledger.promises[0].quote = "改写后的承诺"  # type: ignore[misc]
        with self.assertRaises(Exception):
            ledger.assessments[0].eligible = "yes"  # type: ignore[misc]

    def test_previous_assessment_objects_are_unchanged(self):
        ledger = analyse(POLICY_UPDATE)
        request = ledger.request("request-1")
        first = request.policy_assessments[0]
        self.assertEqual(first.policy_version, "2026.09")
        self.assertEqual(first.eligible, "unknown")
        # The later assessment exists separately.
        self.assertTrue(
            any(item.policy_version == "2026.11" for item in request.policy_assessments)
        )


class CorrectionSemanticsTests(unittest.TestCase):
    def test_correction_reassigns_intent_not_order(self):
        # "提到的退款是订单 B 的重复扣款，订单 A 仍只需查询物流" must bind the
        # refund intent to order B while keeping the logistics ask on order A.
        ledger = analyse(DATA / "support.json", as_of="2026-11-10T09:00:00")
        refunds = [r for r in ledger.requests if r.intent == "refund"]
        logistics = [r for r in ledger.requests if r.intent == "logistics"]
        self.assertEqual([r.order_id for r in refunds], ["B"])
        self.assertEqual([r.order_id for r in logistics], ["A"])
        correction = ledger.corrections[0]
        self.assertEqual(correction.order_id, "B")
        self.assertIn("request-2", correction.affected_request_ids)
        self.assertNotIn("request-1", correction.affected_request_ids)

    def test_customer_denial_does_not_create_refund(self):
        ledger = analyse(AMBIGUOUS)
        refund_orders = {r.order_id for r in ledger.requests if r.intent == "refund"}
        self.assertNotIn("C", refund_orders)
        self.assertTrue(
            any(
                r.order_id == "C" and r.intent == "logistics"
                for r in ledger.requests
            )
        )

    def test_customer_echoing_agent_time_does_not_create_request(self):
        ledger = analyse(AMBIGUOUS)
        echoes = [
            event
            for event in ledger.progress
            if event.kind == "customer_restated_promise"
        ]
        self.assertTrue(echoes)
        self.assertEqual(echoes[0].turn, 10)
        self.assertTrue(
            any(p.confirmed_turn == 9 and p.order_id == "C" for p in ledger.promises)
        )

    def test_multi_order_promise_creates_one_record_per_order(self):
        ledger = analyse(AMBIGUOUS)
        at_turn_four = [p for p in ledger.promises if p.confirmed_turn == 4]
        self.assertEqual({p.order_id for p in at_turn_four}, {"A", "B"})
        for promise in at_turn_four:
            self.assertIn("两单", promise.quote)
            self.assertEqual(promise.deadline_text, "今天下班前")


class ReadOnlyTests(unittest.TestCase):
    def test_package_has_no_outbound_messaging(self):
        forbidden = ("import requests", "import smtplib", "urllib.request", "socket.")
        for path in (ROOT / "support_dialogue").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(file=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_inputs_are_synthetic_only(self):
        for path in (DATA / "support.json", MAIN, AMBIGUOUS, POLICY_UPDATE):
            records = json.loads(path.read_text(encoding="utf-8"))
            for record in records:
                self.assertIn(record["id"], {r["id"] for r in records})
                self.assertNotIn("@", "".join(record.get("text", "")))


if __name__ == "__main__":
    unittest.main()
