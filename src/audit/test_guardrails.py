"""
Payment Retry Sequencer — Phase 7: Guardrail Enforcement Tests
=================================================================

These tests exist to PROVE three specific claims, not just to exercise
the happy path:

  (a) Even if the ML/heuristic layer is adversarially wrong (mocked to
      always recommend an aggressive action), the guardrail wrapper
      still forces no_retry for every hard-stop condition.
  (b) Malformed input never crashes the caller and never silently
      passes through — it always fails safe to no_retry, flagged for
      human review.
  (c) Calling the Phase 3 policy function directly, bypassing the Phase
      7 wrapper entirely, still enforces the hard-stop rules — because
      they're baked into Phase 3's own Layer 1, independently of
      whether Phase 7 is even in the call path.

Plus supporting tests for the audit log's append-only / tamper-evident
properties, since the guardrail guarantees above are only meaningful if
every decision (including overrides and failures) is actually recorded.

Run with:  pytest test_guardrail_enforcement.py -v
"""

from __future__ import annotations

import json

import pytest

from retry_policy_engine import (
    ClassifierOutput,
    MAX_RETRY_ATTEMPTS,
    PolicyDecision,
    RetryAction,
    TransactionContext,
    apply_hard_rules,
    choose_action,
)
from audit_log import AuditLogger
from guardrail_enforcement import (
    enforce_guardrails,
    independent_hard_stop_reason,
    validate_inputs,
    _is_more_aggressive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adversarial_policy_fn(clf, ctx):
    """
    Simulates an arbitrarily wrong / adversarial ML layer: ignores the
    actual inputs entirely and always recommends the most aggressive
    action with maximum confidence, and even lies in its own
    guardrail_checks list claiming everything was already verified.
    This is exactly the kind of output the wrapper must not trust.
    """
    return PolicyDecision(
        recommended_action=RetryAction.SAME_GATEWAY_RETRY.value,
        timing="immediate",
        confidence=0.99,
        reason_string="adversarial model: always retry immediately",
        guardrail_checks=["guardrail:customer_opt_out (not opted out)",
                           "guardrail:max_attempts (ok)"],
        flagged_for_human_review=False,
    )


def _exploding_policy_fn(clf, ctx):
    raise ValueError("simulated ML layer crash")


def _make_ctx(**overrides):
    defaults = dict(
        transaction_id="txn_synthetic_0001",
        attempt_count=1,
        last_attempt_time=None,
        customer_opted_out=False,
        gateway_recent_failure_rate=0.02,
    )
    defaults.update(overrides)
    return TransactionContext(**defaults)


# ---------------------------------------------------------------------------
# (c) Hard-stop rules survive a direct call that bypasses the wrapper
# ---------------------------------------------------------------------------

def test_direct_call_risk_block_bypassing_wrapper_still_blocks():
    """choose_action() called with no wrapper at all must still refuse."""
    clf = ClassifierOutput("risk_block", confidence=0.10, bank_response_code="U99")
    ctx = _make_ctx()
    decision = choose_action(clf, ctx)  # <-- no wrapper involved whatsoever
    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True


def test_direct_call_card_expired_bypassing_wrapper_still_blocks():
    clf = ClassifierOutput("card_expired", confidence=0.99, bank_response_code="U12")
    ctx = _make_ctx()
    decision = choose_action(clf, ctx)
    assert decision.recommended_action == RetryAction.NO_RETRY.value


def test_direct_call_via_apply_hard_rules_layer_only():
    """Exercise Layer 1 directly, one level below choose_action."""
    clf = ClassifierOutput("risk_block", confidence=0.999, bank_response_code="U39")
    ctx = _make_ctx()
    hard = apply_hard_rules(clf, ctx)
    assert hard is not None
    assert hard.recommended_action == RetryAction.NO_RETRY.value


# ---------------------------------------------------------------------------
# (a) Adversarial ML output is overridden by the independent wrapper check
# ---------------------------------------------------------------------------

def test_adversarial_model_overridden_for_risk_block(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_adversarial_policy_fn)

    clf = ClassifierOutput("risk_block", confidence=0.99, bank_response_code="U01")
    ctx = _make_ctx()
    decision = wrapped(clf, ctx)

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True
    assert any("independent_override" in c for c in decision.guardrail_checks)


def test_adversarial_model_overridden_for_risk_coded_raw_value(tmp_path):
    """Category looks benign but the raw bank code is risk-coded."""
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_adversarial_policy_fn)

    clf = ClassifierOutput("bank_server_error", confidence=0.95, bank_response_code="U40")
    ctx = _make_ctx()
    decision = wrapped(clf, ctx)

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True


def test_adversarial_model_overridden_for_card_expired(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_adversarial_policy_fn)

    clf = ClassifierOutput("card_expired", confidence=0.9, bank_response_code="U12")
    ctx = _make_ctx()
    decision = wrapped(clf, ctx)

    assert decision.recommended_action == RetryAction.NO_RETRY.value


def test_adversarial_model_overridden_for_customer_opt_out(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_adversarial_policy_fn)

    # Retry-eligible category, but the customer opted out -- the
    # adversarial fn ignores this; the wrapper must not.
    clf = ClassifierOutput("network_timeout", confidence=0.9, bank_response_code="U71")
    ctx = _make_ctx(customer_opted_out=True)
    decision = wrapped(clf, ctx)

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True


def test_adversarial_model_overridden_for_attempt_cap(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_adversarial_policy_fn)

    clf = ClassifierOutput("network_timeout", confidence=0.9, bank_response_code="U71")
    ctx = _make_ctx(attempt_count=MAX_RETRY_ATTEMPTS)  # at the cap
    decision = wrapped(clf, ctx)

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True


def test_adversarial_model_not_overridden_when_genuinely_safe(tmp_path):
    """
    Sanity check in the other direction: a retry-eligible case with none
    of the hard-stop conditions should NOT be forced to no_retry just
    because the wrapper is present. The wrapper adds safety, it doesn't
    make the system useless.
    """
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_adversarial_policy_fn)

    clf = ClassifierOutput("network_timeout", confidence=0.9, bank_response_code="U71")
    ctx = _make_ctx()  # no opt-out, under the attempt cap
    decision = wrapped(clf, ctx)

    assert decision.recommended_action == RetryAction.SAME_GATEWAY_RETRY.value
    assert decision.flagged_for_human_review is False


# ---------------------------------------------------------------------------
# (b) Malformed input fails safe, never crashes, never passes through
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_clf", [
    None,
    "not a classifier output",
    ClassifierOutput(decline_reason_category="", confidence=0.5, bank_response_code="U01"),
    ClassifierOutput(decline_reason_category="network_timeout", confidence="high", bank_response_code="U01"),
    ClassifierOutput(decline_reason_category="network_timeout", confidence=1.5, bank_response_code="U01"),
    ClassifierOutput(decline_reason_category="network_timeout", confidence=-0.1, bank_response_code="U01"),
    ClassifierOutput(decline_reason_category="network_timeout", confidence=0.5, bank_response_code=42),
])
def test_malformed_classifier_output_fails_safe(tmp_path, bad_clf):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    ctx = _make_ctx()
    decision = wrapped(bad_clf, ctx)  # must not raise

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True
    assert "malformed_input" in decision.guardrail_checks[0]


@pytest.mark.parametrize("bad_ctx_kwargs", [
    dict(attempt_count=-1),
    dict(attempt_count="three"),
    dict(attempt_count=2.5),
    dict(customer_opted_out="yes"),
])
def test_malformed_transaction_context_fails_safe(tmp_path, bad_ctx_kwargs):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    clf = ClassifierOutput("network_timeout", confidence=0.8, bank_response_code="U71")
    ctx = _make_ctx(**bad_ctx_kwargs)
    decision = wrapped(clf, ctx)  # must not raise

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True


def test_completely_wrong_types_do_not_crash(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    decision = wrapped({"category": "network_timeout"}, ["not", "a", "context"])
    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True


def test_exception_inside_wrapped_function_fails_safe(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(_exploding_policy_fn)

    clf = ClassifierOutput("network_timeout", confidence=0.8, bank_response_code="U71")
    ctx = _make_ctx()
    decision = wrapped(clf, ctx)  # must not raise, despite the wrapped fn raising

    assert decision.recommended_action == RetryAction.NO_RETRY.value
    assert decision.flagged_for_human_review is True

    records = list(audit_logger.iter_records())
    assert records[-1]["event_type"] == "policy_fn_exception"


# ---------------------------------------------------------------------------
# Downgrade-only invariant
# ---------------------------------------------------------------------------

def test_is_more_aggressive_ordering():
    assert _is_more_aggressive(
        RetryAction.SAME_GATEWAY_RETRY.value, RetryAction.NO_RETRY.value
    ) is True
    assert _is_more_aggressive(
        RetryAction.NO_RETRY.value, RetryAction.SAME_GATEWAY_RETRY.value
    ) is False
    assert _is_more_aggressive(
        RetryAction.DELAYED_RETRY.value, RetryAction.DELAYED_RETRY.value
    ) is False


def test_wrapper_never_escalates_beyond_wrapped_output(tmp_path):
    """
    The wrapper should never hand back an action riskier than what the
    wrapped function itself returned -- it can only hold steady or move
    toward no_retry.
    """
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    clf = ClassifierOutput("insufficient_funds", confidence=0.9, bank_response_code="U69")
    ctx = _make_ctx()
    raw = choose_action(clf, ctx)
    final = wrapped(clf, ctx)

    from guardrail_enforcement import _rank
    assert _rank(final.recommended_action) <= _rank(raw.recommended_action)


# ---------------------------------------------------------------------------
# Unit tests for the independent-check and validation helpers directly
# ---------------------------------------------------------------------------

def test_independent_hard_stop_reason_covers_all_conditions():
    ctx = _make_ctx()
    assert independent_hard_stop_reason(
        ClassifierOutput("risk_block", 0.1, "U01"), ctx
    ) is not None
    assert independent_hard_stop_reason(
        ClassifierOutput("card_expired", 0.1, "U01"), ctx
    ) is not None
    assert independent_hard_stop_reason(
        ClassifierOutput("network_timeout", 0.1, "U39"), ctx  # risk-coded raw value
    ) is not None
    assert independent_hard_stop_reason(
        ClassifierOutput("network_timeout", 0.1, "U71"), _make_ctx(customer_opted_out=True)
    ) is not None
    assert independent_hard_stop_reason(
        ClassifierOutput("network_timeout", 0.1, "U71"),
        _make_ctx(attempt_count=MAX_RETRY_ATTEMPTS),
    ) is not None
    # genuinely clean case -> no hard stop
    assert independent_hard_stop_reason(
        ClassifierOutput("network_timeout", 0.9, "U71"), ctx
    ) is None


def test_validate_inputs_accepts_well_formed_case():
    clf = ClassifierOutput("network_timeout", 0.9, "U71")
    ctx = _make_ctx()
    assert validate_inputs(clf, ctx) is None


# ---------------------------------------------------------------------------
# Audit log: append-only surface, hash-chain integrity, tamper detection
# ---------------------------------------------------------------------------

def test_audit_logger_exposes_no_edit_or_delete_methods(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    for forbidden in ("delete", "update", "truncate", "remove", "edit", "clear"):
        assert not hasattr(audit_logger, forbidden), (
            f"AuditLogger must not expose a `{forbidden}` method"
        )


def test_audit_log_records_every_call_including_failures(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    wrapped(ClassifierOutput("network_timeout", 0.9, "U71"), _make_ctx())     # ok
    wrapped(ClassifierOutput("risk_block", 0.9, "U01"), _make_ctx())          # hard stop
    wrapped(None, _make_ctx())                                                # malformed

    records = list(audit_logger.iter_records())
    assert len(records) == 3
    assert [r["event_type"] for r in records] == [
        "evaluation", "evaluation", "validation_failure",
    ]


def test_audit_log_chain_is_intact_after_many_writes(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    for i in range(10):
        wrapped(
            ClassifierOutput("network_timeout", 0.9, "U71"),
            _make_ctx(transaction_id=f"txn_{i}"),
        )

    ok, bad_seq = audit_logger.verify_chain()
    assert ok is True
    assert bad_seq is None


def test_audit_log_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit_logger = AuditLogger(str(path))
    wrapped = enforce_guardrails(audit_logger)(choose_action)

    for i in range(5):
        wrapped(
            ClassifierOutput("network_timeout", 0.9, "U71"),
            _make_ctx(transaction_id=f"txn_{i}"),
        )

    ok, _ = audit_logger.verify_chain()
    assert ok is True  # sanity check before tampering

    # Tamper with record #2 directly on disk, outside the AuditLogger API.
    # (Flip it to a value guaranteed to differ from whatever the genuine
    # decision was, so the test can't accidentally "tamper" a value into
    # matching itself.)
    lines = path.read_text().splitlines()
    tampered = json.loads(lines[2])
    original_action = tampered["action_after_guardrails"]
    tampered["action_after_guardrails"] = (
        RetryAction.NO_RETRY.value
        if original_action != RetryAction.NO_RETRY.value
        else RetryAction.SAME_GATEWAY_RETRY.value
    )
    lines[2] = json.dumps(tampered, sort_keys=True, default=str)
    path.write_text("\n".join(lines) + "\n")

    ok, bad_seq = audit_logger.verify_chain()
    assert ok is False
    assert bad_seq == 2


def test_audit_log_survives_and_flags_malformed_events_without_crashing(tmp_path):
    audit_logger = AuditLogger(str(tmp_path / "audit.jsonl"))
    # A non-JSON-serializable-by-default object should still not crash the
    # audit write path because _safe_repr / default=str in AuditLogger
    # falls back gracefully.
    class Weird:
        def __repr__(self):
            return "<Weird object>"

    record = audit_logger.record({
        "timestamp": "2026-01-01T00:00:00+00:00",
        "model_version": "test",
        "event_type": "evaluation",
        "input_features": {"note": Weird()},
        "action_before_guardrails": "same_gateway_retry",
        "action_after_guardrails": "no_retry",
        "guardrails_fired": [],
    })
    assert record["seq"] == 0
    assert "record_hash" in record
