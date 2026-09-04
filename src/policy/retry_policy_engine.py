"""
Payment Retry Sequencer — Phase 3: Retry Policy Engine
=======================================================

Consumes the Phase 2 classifier's output (decline_reason_category + confidence
+ raw bank_response_code) and decides what to do next.

Architecture (three layers, strictly ordered):

    1. HARD RULE TABLE   (apply_hard_rules)   — deterministic, non-ML,
       cannot be overridden by anything downstream.
    2. ML / HEURISTIC LAYER (RecoveryEstimator, choose_action) — for
       retry-eligible categories only, estimates P(recovery | action)
       per candidate action and picks the best one by expected value.
    3. GUARDRAILS        (apply_guardrails)   — runs AFTER step 2, can
       only make the outcome MORE conservative (downgrade), never less.

Why this shape: Phase 2 measured an ~87.2% macro-F1 classifier with a
3.6% dangerous-misclassification rate on risk_block and a 3.2% rate on
card_expired (true risk/expired cases the model itself routed toward a
retryable-looking category). That's a good classifier and still not
something you let gate an irreversible, safety-critical decision on its
own. See the design note at the bottom of this file for the full
reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------------

MAX_RETRY_ATTEMPTS = 3
LOW_CONFIDENCE_THRESHOLD = 0.50  # below this, don't trust the classifier's
                                 # category call enough to fire an immediate
                                 # gateway retry off of it


class RetryAction(str, Enum):
    NO_RETRY = "no_retry"
    SAME_GATEWAY_RETRY = "same_gateway_retry"
    ALT_GATEWAY_RETRY = "alt_gateway_retry"
    ALT_PAYMENT_METHOD_PROMPT = "alt_payment_method_prompt"
    DELAYED_RETRY = "delayed_retry"


COOLDOWN_SECONDS = {
    RetryAction.SAME_GATEWAY_RETRY.value: 30,
    RetryAction.ALT_GATEWAY_RETRY.value: 15,
    RetryAction.ALT_PAYMENT_METHOD_PROMPT.value: 0,   # customer-driven
    RetryAction.DELAYED_RETRY.value: 0,               # timing already encodes the wait
}

# Categories the Phase 2 model can emit that are hard-gated to no_retry
# regardless of confidence. Everything else falls through to the ML layer.
NO_RETRY_CATEGORIES = frozenset({"risk_block", "card_expired"})

# Independent cross-check per Phase 2's own recommendation: a raw
# bank_response_code that Phase 1's generator reserves for risk blocks
# forces the no_retry path even if the classifier's *category* output
# says something else. Replace this placeholder set with the real
# risk-coded values from Phase 1's generator config before relying on it.
RISK_CODED_RAW_VALUES = frozenset({
    "U39", "U40", "U41",  # PLACEHOLDER — fill in from Phase 1 config
})

# ---------------------------------------------------------------------------
# Layer 2 priors: P(recovery | action) for retry-eligible categories.
#
# These are documented starting points based on domain reasoning about
# *why* each decline happens (e.g. a funds shortfall doesn't get fixed by
# hitting a different gateway; a flaky gateway often does recover on its
# own or on an alternate rail). They are NOT fit to real retry_outcome
# data, because Phase 1/2 didn't generate that column. Replace via
# RecoveryEstimator.from_trained_model() once real outcome history exists;
# the .score() interface is designed to stay stable across that swap.
# ---------------------------------------------------------------------------

BASE_RECOVERY_PRIORS: Dict[str, Dict[RetryAction, float]] = {
    "insufficient_funds": {
        RetryAction.SAME_GATEWAY_RETRY: 0.04,
        RetryAction.ALT_GATEWAY_RETRY: 0.04,
        RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.55,
        RetryAction.DELAYED_RETRY: 0.30,
    },
    "bank_server_error": {
        RetryAction.SAME_GATEWAY_RETRY: 0.62,
        RetryAction.ALT_GATEWAY_RETRY: 0.58,
        RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.20,
        RetryAction.DELAYED_RETRY: 0.68,
    },
    "invalid_otp": {
        RetryAction.SAME_GATEWAY_RETRY: 0.58,
        RetryAction.ALT_GATEWAY_RETRY: 0.20,
        RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.35,
        RetryAction.DELAYED_RETRY: 0.30,
    },
    "network_timeout": {
        RetryAction.SAME_GATEWAY_RETRY: 0.70,
        RetryAction.ALT_GATEWAY_RETRY: 0.65,
        RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.15,
        RetryAction.DELAYED_RETRY: 0.60,
    },
    "limit_exceeded": {
        RetryAction.SAME_GATEWAY_RETRY: 0.03,
        RetryAction.ALT_GATEWAY_RETRY: 0.03,
        RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.50,
        RetryAction.DELAYED_RETRY: 0.35,
    },
    "other": {
        RetryAction.SAME_GATEWAY_RETRY: 0.25,
        RetryAction.ALT_GATEWAY_RETRY: 0.20,
        RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.25,
        RetryAction.DELAYED_RETRY: 0.25,
    },
}

# Expected-value bookkeeping: value of a successful recovery is normalized
# to 1.0; each action carries a small cost when it fails (gateway fees,
# customer friction, infra load). This keeps the policy from chasing a
# near-zero-probability retry just because it's nominally "free."
VALUE_OF_RECOVERY = 1.0
COST_OF_FAILED_ATTEMPT: Dict[RetryAction, float] = {
    RetryAction.SAME_GATEWAY_RETRY: 0.05,
    RetryAction.ALT_GATEWAY_RETRY: 0.08,
    RetryAction.ALT_PAYMENT_METHOD_PROMPT: 0.02,
    RetryAction.DELAYED_RETRY: 0.03,
}

DEFAULT_TIMING = {
    RetryAction.SAME_GATEWAY_RETRY: "immediate",
    RetryAction.ALT_GATEWAY_RETRY: "immediate",
    RetryAction.ALT_PAYMENT_METHOD_PROMPT: "immediate",  # customer acts now
    RetryAction.DELAYED_RETRY: None,  # computed per-category below
}

DELAYED_RETRY_WAIT: Dict[str, timedelta] = {
    "insufficient_funds": timedelta(hours=12),
    "bank_server_error": timedelta(minutes=5),
    "invalid_otp": timedelta(minutes=2),
    "network_timeout": timedelta(minutes=2),
    "limit_exceeded": timedelta(hours=24),
    "other": timedelta(hours=6),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ClassifierOutput:
    """Phase 2 model output for one failed attempt."""
    decline_reason_category: str
    confidence: float
    bank_response_code: str


@dataclass
class TransactionContext:
    """Everything the policy engine needs about the transaction/customer."""
    transaction_id: str
    attempt_count: int                       # attempts so far, including this failed one
    last_attempt_time: Optional[datetime] = None
    customer_opted_out: bool = False
    gateway_recent_failure_rate: Optional[float] = None
    merchant_hist_failure_rate: Optional[float] = None
    amount: Optional[float] = None
    payment_method: Optional[str] = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PolicyDecision:
    recommended_action: str
    timing: Optional[str]
    confidence: float
    reason_string: str
    guardrail_checks: List[str]
    flagged_for_human_review: bool = False

    def to_dict(self) -> dict:
        return {
            "recommended_action": self.recommended_action,
            "timing": self.timing,
            "confidence": self.confidence,
            "reason_string": self.reason_string,
            "guardrail_checks": self.guardrail_checks,
            "flagged_for_human_review": self.flagged_for_human_review,
        }


# ---------------------------------------------------------------------------
# Layer 1: Hard rule table (non-ML, cannot be overridden)
# ---------------------------------------------------------------------------

def apply_hard_rules(clf: ClassifierOutput, ctx: TransactionContext) -> Optional[PolicyDecision]:
    """
    Deterministic safety layer, evaluated before any ML/heuristic scoring.

    Returns a final PolicyDecision if the case is hard-gated (risk_block /
    card_expired, by category OR by an independent raw-code cross-check),
    or None if the case is retry-eligible and should proceed to Layer 2.

    Nothing downstream can change what this function decides.
    """
    category = clf.decline_reason_category
    raw_code_says_risk = clf.bank_response_code in RISK_CODED_RAW_VALUES

    if category == "risk_block" or raw_code_says_risk:
        if category == "risk_block":
            trigger = "classifier category = risk_block"
        else:
            trigger = (
                f"raw bank_response_code '{clf.bank_response_code}' is in the "
                f"risk-coded set, overriding classifier category '{category}'"
            )
        return PolicyDecision(
            recommended_action=RetryAction.NO_RETRY.value,
            timing=None,
            confidence=1.0,  # rule-based certainty, not a model probability
            reason_string=(
                f"HARD RULE: {trigger} -> no_retry, mandatory human review. "
                f"No alternate gateway or payment method is suggested for "
                f"this case."
            ),
            guardrail_checks=["hard_rule:risk_block_no_retry"],
            flagged_for_human_review=True,
        )

    if category == "card_expired":
        return PolicyDecision(
            recommended_action=RetryAction.NO_RETRY.value,
            timing=None,
            confidence=1.0,
            reason_string=(
                "HARD RULE: card_expired -> no_retry. Customer must supply "
                "new card details; not eligible for an automated retry."
            ),
            guardrail_checks=["hard_rule:card_expired_no_retry"],
            flagged_for_human_review=False,
        )

    return None  # retry-eligible, continue to Layer 2


# ---------------------------------------------------------------------------
# Layer 2: ML / heuristic recovery-probability estimator
# ---------------------------------------------------------------------------

def _adjust_for_gateway_health(
    category: str, priors: Dict[RetryAction, float], ctx: TransactionContext
) -> Dict[RetryAction, float]:
    """Down-weight same-gateway retry when the gateway is currently unhealthy."""
    adjusted = dict(priors)
    if (
        ctx.gateway_recent_failure_rate is not None
        and category in ("bank_server_error", "network_timeout")
    ):
        penalty = min(ctx.gateway_recent_failure_rate, 0.9) * 0.5
        adjusted[RetryAction.SAME_GATEWAY_RETRY] = max(
            0.01, adjusted[RetryAction.SAME_GATEWAY_RETRY] - penalty
        )
    return adjusted


class RecoveryEstimator:
    """
    Estimates P(recovery | action) for each candidate retry action, for
    decline categories that have already passed Layer 1 (risk_block and
    card_expired never reach this class).

    Deliberately simple and auditable rather than "fancy," per the brief.
    Swap in a real trained model via `from_trained_model()` once
    retry_outcome history exists — the `.score()` interface is the
    contract the rest of the policy engine depends on, so nothing else
    needs to change when you do.
    """

    def __init__(self, priors: Optional[Dict[str, Dict[RetryAction, float]]] = None):
        self.priors = priors or BASE_RECOVERY_PRIORS

    def score(self, category: str, ctx: TransactionContext) -> Dict[RetryAction, float]:
        if category not in self.priors:
            raise ValueError(
                f"No recovery-probability prior for category '{category}'. "
                f"Only retry-eligible categories should ever reach "
                f"RecoveryEstimator — check that apply_hard_rules() ran first."
            )
        return _adjust_for_gateway_health(category, self.priors[category], ctx)

    @classmethod
    def from_trained_model(cls, model, feature_fn):
        """
        Production path: wrap a trained model (e.g. one calibrated
        probability head per action, or four one-vs-rest classifiers,
        fit on real retry_outcome history) behind the same .score()
        interface used above.
        """
        raise NotImplementedError(
            "Plug in a trained model here once retry_outcome data exists. "
            "Preserve the .score(category, ctx) -> Dict[RetryAction, float] "
            "signature so choose_action() below needs no changes."
        )


def choose_action(
    clf: ClassifierOutput,
    ctx: TransactionContext,
    estimator: Optional[RecoveryEstimator] = None,
) -> PolicyDecision:
    """Runs Layer 1, then (if eligible) Layer 2, then Layer 3 guardrails."""
    hard = apply_hard_rules(clf, ctx)
    if hard is not None:
        return hard

    estimator = estimator or RecoveryEstimator()
    category = clf.decline_reason_category
    probs = estimator.score(category, ctx)

    expected_value = {
        action: p * VALUE_OF_RECOVERY - (1 - p) * COST_OF_FAILED_ATTEMPT[action]
        for action, p in probs.items()
    }
    best_action, best_ev = max(expected_value.items(), key=lambda kv: kv[1])

    if best_ev <= 0:
        # No candidate action beats the implicit "do nothing" baseline
        # (EV = 0: no attempt, no recovery, no attempt cost).
        decision = PolicyDecision(
            recommended_action=RetryAction.NO_RETRY.value,
            timing=None,
            confidence=round(1 - max(probs.values()), 3),
            reason_string=(
                f"ML layer: no candidate action for '{category}' clears the "
                f"no-retry baseline (best expected value {best_ev:.3f}, from "
                f"{best_action.value}). Recommending no_retry — this is a "
                f"soft, revisitable call, unlike the Layer-1 hard no_retry."
            ),
            guardrail_checks=[],
            flagged_for_human_review=False,
        )
    else:
        timing = DEFAULT_TIMING[best_action]
        if best_action == RetryAction.DELAYED_RETRY:
            wait = DELAYED_RETRY_WAIT.get(category, timedelta(hours=6))
            timing = f"delay_{int(wait.total_seconds())}s"
        prob_summary = ", ".join(f"{a.value}={p:.2f}" for a, p in probs.items())
        decision = PolicyDecision(
            recommended_action=best_action.value,
            timing=timing,
            confidence=round(probs[best_action], 3),
            reason_string=(
                f"ML/heuristic layer: category='{category}' "
                f"(classifier confidence {clf.confidence:.2f}) -> "
                f"{best_action.value} has the highest estimated recovery "
                f"probability ({probs[best_action]:.2f}). Candidates: "
                f"{prob_summary}."
            ),
            guardrail_checks=[],
            flagged_for_human_review=False,
        )

    return apply_guardrails(decision, clf, ctx)


# ---------------------------------------------------------------------------
# Layer 3: Guardrails — downgrade-only wrapper, runs after the ML suggestion
# ---------------------------------------------------------------------------

def apply_guardrails(
    decision: PolicyDecision, clf: ClassifierOutput, ctx: TransactionContext
) -> PolicyDecision:
    """
    Runs after Layer 1/2. Can only move the outcome to something SAFER:
    - shrink an aggressive action to a gentler one
    - push an immediate action back to a delayed one
    - collapse anything to no_retry

    Never turns a no_retry into a retry, and never turns a conservative
    action into a more aggressive one. Layer 1's no_retry decisions pass
    through untouched — every check below is a no-op once the action is
    already no_retry, since that's the floor.
    """
    checks = list(decision.guardrail_checks)
    action = decision.recommended_action
    timing = decision.timing
    confidence = decision.confidence
    flagged = decision.flagged_for_human_review
    notes: List[str] = []

    # 1. Customer opt-out — absolute stop.
    if ctx.customer_opted_out:
        if action != RetryAction.NO_RETRY.value:
            action = RetryAction.NO_RETRY.value
            timing = None
            notes.append("customer has opted out of automated retries -> no_retry")
        checks.append("guardrail:customer_opt_out")
    else:
        checks.append("guardrail:customer_opt_out (not opted out)")

    # 2. Max retry attempts cap.
    if action != RetryAction.NO_RETRY.value and ctx.attempt_count >= MAX_RETRY_ATTEMPTS:
        action = RetryAction.NO_RETRY.value
        timing = None
        flagged = True
        notes.append(
            f"attempt_count={ctx.attempt_count} reached cap of "
            f"{MAX_RETRY_ATTEMPTS} -> no_retry, flagged for review"
        )
        checks.append(f"guardrail:max_attempts (count={ctx.attempt_count}, cap={MAX_RETRY_ATTEMPTS})")
    else:
        checks.append(f"guardrail:max_attempts (count={ctx.attempt_count}, cap={MAX_RETRY_ATTEMPTS}, ok)")

    # 3. Cooldown window between retries — push an immediate action back
    #    to a delayed one if we're firing too soon after the last attempt.
    if (
        action not in (RetryAction.NO_RETRY.value, RetryAction.DELAYED_RETRY.value)
        and ctx.last_attempt_time is not None
    ):
        cooldown = COOLDOWN_SECONDS.get(action, 30)
        elapsed = (ctx.now - ctx.last_attempt_time).total_seconds()
        if elapsed < cooldown:
            remaining = int(cooldown - elapsed)
            action = RetryAction.DELAYED_RETRY.value
            timing = f"delay_{remaining}s"
            notes.append(
                f"only {elapsed:.0f}s elapsed since last attempt (cooldown "
                f"{cooldown}s) -> downgraded to delayed_retry"
            )
        checks.append(f"guardrail:cooldown_window (elapsed={elapsed:.0f}s, required={cooldown}s)")
    else:
        checks.append("guardrail:cooldown_window (n/a)")

    # 4. Low classifier-confidence downgrade — defense in depth against a
    #    Phase 2 category call the model itself wasn't confident about.
    #    NOTE: this only gates same_gateway_retry/alt_gateway_retry (the two
    #    "aggressive" immediate actions). If the ML layer already picked
    #    delayed_retry or alt_payment_method_prompt on its own merits, low
    #    confidence doesn't trigger anything further here — those actions
    #    are already the conservative choice, so there's nothing to downgrade.
    if (
        action in (RetryAction.SAME_GATEWAY_RETRY.value, RetryAction.ALT_GATEWAY_RETRY.value)
        and clf.confidence < LOW_CONFIDENCE_THRESHOLD
    ):
        wait = DELAYED_RETRY_WAIT.get(clf.decline_reason_category, timedelta(hours=6))
        action = RetryAction.DELAYED_RETRY.value
        timing = f"delay_{int(wait.total_seconds())}s"
        notes.append(
            f"classifier confidence {clf.confidence:.2f} below threshold "
            f"{LOW_CONFIDENCE_THRESHOLD} -> downgraded immediate gateway "
            f"retry to delayed_retry"
        )
        checks.append(f"guardrail:low_classifier_confidence (confidence={clf.confidence:.2f})")
    else:
        checks.append(f"guardrail:low_classifier_confidence (confidence={clf.confidence:.2f}, ok)")

    reason = decision.reason_string
    if notes:
        reason = reason + " || Guardrails applied: " + "; ".join(notes)

    return PolicyDecision(
        recommended_action=action,
        timing=timing,
        confidence=confidence,
        reason_string=reason,
        guardrail_checks=checks,
        flagged_for_human_review=flagged,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_retry_policy(
    clf: ClassifierOutput,
    ctx: TransactionContext,
    estimator: Optional[RecoveryEstimator] = None,
) -> dict:
    """Main entry point: classifier output + context -> policy decision dict."""
    return choose_action(clf, ctx, estimator).to_dict()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    now = datetime.now(timezone.utc)
    scenarios = [
        (
            "risk_block, HIGH confidence retry-shaped classifier score "
            "-> still hard no_retry",
            ClassifierOutput("risk_block", confidence=0.98, bank_response_code="U39"),
            TransactionContext("txn_001", attempt_count=1, last_attempt_time=None, now=now),
        ),
        (
            "bank_server_error category, but raw code is risk-coded "
            "-> cross-check overrides to no_retry",
            ClassifierOutput("bank_server_error", confidence=0.81, bank_response_code="U40"),
            TransactionContext("txn_002", attempt_count=1, last_attempt_time=None, now=now),
        ),
        (
            "card_expired -> no_retry, prompt for new card",
            ClassifierOutput("card_expired", confidence=0.95, bank_response_code="U12"),
            TransactionContext("txn_003", attempt_count=1, last_attempt_time=None, now=now),
        ),
        (
            "network_timeout, healthy gateway, first attempt -> retry",
            ClassifierOutput("network_timeout", confidence=0.88, bank_response_code="U71"),
            TransactionContext(
                "txn_004", attempt_count=1, last_attempt_time=None,
                gateway_recent_failure_rate=0.02, now=now,
            ),
        ),
        (
            "network_timeout, last attempt 5s ago -> cooldown downgrades to delayed",
            ClassifierOutput("network_timeout", confidence=0.88, bank_response_code="U71"),
            TransactionContext(
                "txn_005", attempt_count=2, last_attempt_time=now - timedelta(seconds=5),
                gateway_recent_failure_rate=0.02, now=now,
            ),
        ),
        (
            "network_timeout, low classifier confidence -> ML layer's own "
            "pick (same_gateway_retry) gets downgraded to delayed by the "
            "guardrail, not by the ML scoring itself",
            ClassifierOutput("network_timeout", confidence=0.30, bank_response_code="U71"),
            TransactionContext(
                "txn_006", attempt_count=1, last_attempt_time=None,
                gateway_recent_failure_rate=0.02, now=now,
            ),
        ),
        (
            "insufficient_funds, already at attempt cap -> no_retry, flagged",
            ClassifierOutput("insufficient_funds", confidence=0.92, bank_response_code="U69"),
            TransactionContext("txn_007", attempt_count=3, last_attempt_time=None, now=now),
        ),
        (
            "limit_exceeded, customer opted out -> no_retry regardless of ML score",
            ClassifierOutput("limit_exceeded", confidence=0.90, bank_response_code="U51"),
            TransactionContext(
                "txn_008", attempt_count=1, last_attempt_time=None,
                customer_opted_out=True, now=now,
            ),
        ),
    ]

    for label, clf, ctx in scenarios:
        print(f"\n=== {label} ===")
        print(json.dumps(evaluate_retry_policy(clf, ctx), indent=2))
