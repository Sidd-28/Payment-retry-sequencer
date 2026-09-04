"""
Payment Retry Sequencer — Phase 7: Guardrail Enforcement + Audit Wrapper
=========================================================================

Wraps any Phase 3 policy-decision function (e.g. `choose_action`) so that
EVERY call, no matter how it's invoked or what the wrapped function
returns, gets three guarantees before anything reaches the caller:

  1. Malformed input never reaches the wrapped function or crashes the
     caller. It is rejected up front and fails SAFE to no_retry, flagged
     for human review.

  2. The hard-stop conditions (risk_block, risk-coded raw response,
     card_expired, customer opt-out, attempt cap) are re-derived
     independently from the raw `clf`/`ctx` inputs — NOT read off of
     whatever the wrapped function's own PolicyDecision claims. This is
     the crux of the defense-in-depth story: if the wrapped function is
     buggy, mocked, or adversarially manipulated to always recommend an
     aggressive action, this layer still forces no_retry, because it
     never trusts the wrapped function's conclusion in the first place.

  3. The wrapper can only make the final action safer than (or equal
     to) what the wrapped function returned — never more aggressive.
     This is asserted explicitly (see `_is_more_aggressive`), not just
     assumed from the logic above.

Every call is written to the append-only AuditLogger before the decision
is returned — including validation failures and exceptions raised by the
wrapped function, so the audit trail has no gaps.

IMPORTANT: this module does not modify Phase 3 (`retry_policy_engine.py`)
at all. Phase 3's own Layer 1 hard-rule table and Layer 3 guardrails
still run when `choose_action()` is called directly, with or without
this wrapper. This wrapper is an *additional*, independent layer on top
— not a replacement for Phase 3's own checks.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from retry_policy_engine import (
    ClassifierOutput,
    MAX_RETRY_ATTEMPTS,
    PolicyDecision,
    RetryAction,
    RISK_CODED_RAW_VALUES,
    TransactionContext,
)
from audit_log import AuditLogger

logger = logging.getLogger("retry_sequencer.guardrail_enforcement")

DEFAULT_MODEL_VERSION = "phase3-policy-engine-v1"

# Coarse "how aggressive is this action" ranking used ONLY to assert the
# downgrade-only invariant in step 4 of the wrapper below. no_retry is the
# floor (0); the two immediate-gateway-hit actions are the most aggressive.
_ACTION_RISK_RANK = {
    RetryAction.NO_RETRY.value: 0,
    RetryAction.DELAYED_RETRY.value: 1,
    RetryAction.ALT_PAYMENT_METHOD_PROMPT.value: 1,
    RetryAction.ALT_GATEWAY_RETRY.value: 2,
    RetryAction.SAME_GATEWAY_RETRY.value: 2,
}


def _rank(action: Optional[str]) -> int:
    # Unknown/garbage action strings are treated as maximally aggressive
    # so that anything we don't recognize gets clamped rather than trusted.
    return _ACTION_RISK_RANK.get(action, 99)


def _is_more_aggressive(candidate: Optional[str], baseline: Optional[str]) -> bool:
    return _rank(candidate) > _rank(baseline)


# ---------------------------------------------------------------------------
# Independent hard-stop re-derivation (does not trust the wrapped decision)
# ---------------------------------------------------------------------------

def independent_hard_stop_reason(clf, ctx) -> Optional[str]:
    """
    Re-derive, directly from the raw inputs, whether this case MUST be
    no_retry — independently of whatever PolicyDecision the wrapped
    function produced. Mirrors Phase 3's own hard-rule conditions, but
    is deliberately re-implemented here rather than imported as a
    boolean result, so a bug in Phase 3's decision object can't silently
    disable this check.

    Returns a human-readable reason string if a hard stop applies, else
    None. Never raises: any attribute-access failure is treated as "not
    enough information to clear this case" and is folded into the
    caller's malformed-input handling instead.
    """
    category = getattr(clf, "decline_reason_category", None)
    raw_code = getattr(clf, "bank_response_code", None)

    if category == "risk_block":
        return "independent check: decline_reason_category == 'risk_block'"
    if raw_code in RISK_CODED_RAW_VALUES:
        return f"independent check: bank_response_code {raw_code!r} is risk-coded"
    if category == "card_expired":
        return "independent check: decline_reason_category == 'card_expired'"
    if getattr(ctx, "customer_opted_out", False) is True:
        return "independent check: customer_opted_out is True"

    attempt_count = getattr(ctx, "attempt_count", None)
    if isinstance(attempt_count, int) and attempt_count >= MAX_RETRY_ATTEMPTS:
        return (
            f"independent check: attempt_count={attempt_count} >= "
            f"cap {MAX_RETRY_ATTEMPTS}"
        )
    return None


# ---------------------------------------------------------------------------
# Input validation — fail-safe, never raises
# ---------------------------------------------------------------------------

def validate_inputs(clf, ctx) -> Optional[str]:
    """Returns an error description if inputs are structurally unsound, else None."""
    if not isinstance(clf, ClassifierOutput):
        return f"clf is not a ClassifierOutput (got {type(clf).__name__})"
    if not isinstance(ctx, TransactionContext):
        return f"ctx is not a TransactionContext (got {type(ctx).__name__})"
    if not isinstance(clf.decline_reason_category, str) or not clf.decline_reason_category:
        return f"decline_reason_category missing or not a non-empty string: {clf.decline_reason_category!r}"
    if isinstance(clf.confidence, bool) or not isinstance(clf.confidence, (int, float)):
        return f"confidence is not numeric: {clf.confidence!r}"
    if not (0.0 <= float(clf.confidence) <= 1.0):
        return f"confidence out of [0, 1] range: {clf.confidence!r}"
    if not isinstance(clf.bank_response_code, str):
        return f"bank_response_code is not a string: {clf.bank_response_code!r}"
    if isinstance(ctx.attempt_count, bool) or not isinstance(ctx.attempt_count, int):
        return f"attempt_count is not an int: {ctx.attempt_count!r}"
    if ctx.attempt_count < 0:
        return f"attempt_count is negative: {ctx.attempt_count!r}"
    if not isinstance(ctx.customer_opted_out, bool):
        return f"customer_opted_out is not a bool: {ctx.customer_opted_out!r}"
    return None


# ---------------------------------------------------------------------------
# Audit serialization helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    return str(o)


def _safe_repr(clf, ctx) -> dict:
    """
    Best-effort, never-raises serialization of the raw inputs for the
    audit record. Falls back to repr() for anything that isn't a plain
    dataclass, so a malformed/mocked clf or ctx can never prevent the
    audit write itself.
    """
    def _dump(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return repr(obj)

    try:
        raw = {"clf": _dump(clf), "ctx": _dump(ctx)}
        return json.loads(json.dumps(raw, default=_json_default))
    except Exception as exc:  # pragma: no cover - last-resort fallback
        return {"clf": repr(clf), "ctx": repr(ctx), "serialization_error": str(exc)}


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------

def enforce_guardrails(
    audit_logger: AuditLogger,
    model_version: str = DEFAULT_MODEL_VERSION,
):
    """
    Decorator factory. Wrap any `(clf, ctx, ...) -> PolicyDecision`
    function (e.g. `retry_policy_engine.choose_action`) so that every
    call is guardrail-checked and audit-logged before returning.

    Usage:
        audit_logger = AuditLogger("audit_log.jsonl")
        safe_choose_action = enforce_guardrails(audit_logger)(choose_action)
        decision = safe_choose_action(clf, ctx)
    """

    def decorator(policy_fn: Callable[..., PolicyDecision]):
        @functools.wraps(policy_fn)
        def wrapper(clf, ctx, *args, **kwargs) -> PolicyDecision:
            timestamp = datetime.now(timezone.utc).isoformat()

            # --- Step 1: malformed input -> fail safe before calling anything ---
            validation_error = validate_inputs(clf, ctx)
            if validation_error is not None:
                decision = PolicyDecision(
                    recommended_action=RetryAction.NO_RETRY.value,
                    timing=None,
                    confidence=0.0,
                    reason_string=(
                        "GUARDRAIL WRAPPER: malformed input rejected before "
                        "reaching the policy function -> no_retry, flagged "
                        f"for human review. Detail: {validation_error}"
                    ),
                    guardrail_checks=["guardrail_wrapper:malformed_input"],
                    flagged_for_human_review=True,
                )
                _safe_audit(audit_logger, {
                    "timestamp": timestamp,
                    "model_version": model_version,
                    "event_type": "validation_failure",
                    "input_features": _safe_repr(clf, ctx),
                    "decline_reason_category": getattr(clf, "decline_reason_category", None),
                    "confidence": getattr(clf, "confidence", None),
                    "action_before_guardrails": None,
                    "action_after_guardrails": decision.recommended_action,
                    "guardrails_fired": decision.guardrail_checks,
                    "flagged_for_human_review": True,
                    "detail": validation_error,
                })
                return decision

            # --- Step 2: run the wrapped policy function; never let it crash the caller ---
            try:
                raw_decision = policy_fn(clf, ctx, *args, **kwargs)
                if not isinstance(raw_decision, PolicyDecision):
                    raise TypeError(
                        f"policy function returned {type(raw_decision).__name__}, "
                        f"expected PolicyDecision"
                    )
            except Exception as exc:
                decision = PolicyDecision(
                    recommended_action=RetryAction.NO_RETRY.value,
                    timing=None,
                    confidence=0.0,
                    reason_string=(
                        f"GUARDRAIL WRAPPER: policy function raised "
                        f"{type(exc).__name__}: {exc} -> fail-safe no_retry, "
                        f"flagged for human review."
                    ),
                    guardrail_checks=["guardrail_wrapper:policy_fn_exception"],
                    flagged_for_human_review=True,
                )
                _safe_audit(audit_logger, {
                    "timestamp": timestamp,
                    "model_version": model_version,
                    "event_type": "policy_fn_exception",
                    "input_features": _safe_repr(clf, ctx),
                    "decline_reason_category": getattr(clf, "decline_reason_category", None),
                    "confidence": getattr(clf, "confidence", None),
                    "action_before_guardrails": None,
                    "action_after_guardrails": decision.recommended_action,
                    "guardrails_fired": decision.guardrail_checks,
                    "flagged_for_human_review": True,
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                return decision

            action_before = raw_decision.recommended_action
            fired = list(raw_decision.guardrail_checks)

            # --- Step 3: independent hard-stop re-check (does not trust raw_decision) ---
            hard_stop_reason = independent_hard_stop_reason(clf, ctx)
            if hard_stop_reason and action_before != RetryAction.NO_RETRY.value:
                final_decision = dataclasses.replace(
                    raw_decision,
                    recommended_action=RetryAction.NO_RETRY.value,
                    timing=None,
                    flagged_for_human_review=True,
                    reason_string=(
                        raw_decision.reason_string
                        + f" || GUARDRAIL WRAPPER OVERRIDE: {hard_stop_reason} "
                          f"-> forced no_retry regardless of the policy "
                          f"function's own output."
                    ),
                    guardrail_checks=fired + [
                        f"guardrail_wrapper:independent_override ({hard_stop_reason})"
                    ],
                )
            elif hard_stop_reason:
                final_decision = dataclasses.replace(
                    raw_decision,
                    guardrail_checks=fired + [
                        f"guardrail_wrapper:independent_check_confirmed ({hard_stop_reason})"
                    ],
                )
            else:
                final_decision = dataclasses.replace(
                    raw_decision,
                    guardrail_checks=fired + ["guardrail_wrapper:independent_check_passed"],
                )

            # --- Step 4: explicit downgrade-only invariant (belt-and-suspenders) ---
            if _is_more_aggressive(final_decision.recommended_action, action_before):
                # Structurally this branch should be unreachable given Step 3
                # only ever moves the action toward no_retry — but we assert
                # it explicitly rather than relying on that being true forever.
                final_decision = dataclasses.replace(
                    final_decision,
                    recommended_action=RetryAction.NO_RETRY.value,
                    timing=None,
                    flagged_for_human_review=True,
                    reason_string=(
                        final_decision.reason_string
                        + " || GUARDRAIL WRAPPER INVARIANT TRIP: refused to "
                          "return an action more aggressive than the policy "
                          "function's own output -> forced no_retry."
                    ),
                )

            _safe_audit(audit_logger, {
                "timestamp": timestamp,
                "model_version": model_version,
                "event_type": "evaluation",
                "input_features": _safe_repr(clf, ctx),
                "decline_reason_category": clf.decline_reason_category,
                "confidence": clf.confidence,
                "action_before_guardrails": action_before,
                "action_after_guardrails": final_decision.recommended_action,
                "guardrails_fired": final_decision.guardrail_checks,
                "flagged_for_human_review": final_decision.flagged_for_human_review,
            })

            return final_decision

        return wrapper

    return decorator


def _safe_audit(audit_logger: AuditLogger, event: dict) -> None:
    """
    Write to the audit log without letting a logging outage take down
    the (already-computed, safe) decision path. A failure here is
    surfaced loudly via `logging` so it can page an operator — in a
    production deployment this should be wired to real alerting, since
    a gap in the audit trail is itself a compliance incident even
    though it doesn't change what action was returned to the caller.
    """
    try:
        audit_logger.record(event)
    except Exception:
        logger.error("AUDIT LOG WRITE FAILED for event: %s", event, exc_info=True)
