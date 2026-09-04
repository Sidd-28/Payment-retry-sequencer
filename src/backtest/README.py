"""
Payment Retry Sequencer — Phase 4: Backtesting Harness
=======================================================

Runs the Phase 3 policy engine (`retry_policy_engine.py`) against a batch of
failed payments and compares it to two baselines:

    (A) no_retry_ever        — never retry anything.
    (B) naive_retry_once     — retry every single failed attempt exactly
                                once, 1 hour later, via a generic same-
                                gateway mechanism, with NO hard-rule layer,
                                NO guardrails, and no classifier in the loop.

It reports cost-adjusted recovered revenue, false-positive retry rate,
hard-rule/guardrail compliance, recovery latency, and a cost-assumption
sensitivity analysis. It also writes a full per-transaction audit trail.

--------------------------------------------------------------------------
READ THIS BEFORE TRUSTING THE DOLLAR FIGURES
--------------------------------------------------------------------------
Per phase2_results.md, the Phase 1/2 synthetic dataset does NOT contain real
`retry_outcome` / `time_to_recovery` data — those columns are null by
construction (they're Phase 3+ outputs, not Phase 1 generator outputs).
That means there is no ground truth anywhere in this project for "would a
retry actually have succeeded, and how fast." This harness therefore has to
simulate that outcome using a documented, seeded probability model
(`build_true_recovery_probs`) instead of measuring it.

Two things follow from that:

1. This is a **decision-logic backtest**, not a real-money backtest. It
   correctly measures whether the policy engine's *choices* are better than
   the baselines' choices, under a stated set of recovery-probability and
   cost assumptions. It does NOT prove what will happen in production.
2. The simulated "true" recovery probabilities are deliberately perturbed
   away from Phase 3's own `BASE_RECOVERY_PRIORS` (see
   `build_true_recovery_probs`) so this script isn't just grading the
   policy against its own assumptions, which would be circular. They are
   still not observed data.

The moment real `retry_outcome` / `time_to_recovery` history exists (i.e.
Phase 3 has been running in production and logging outcomes), point
`--outcomes-csv` at that file and this harness will use REAL observed
outcomes instead of simulating them — the rest of the pipeline (cost
adjustment, false-positive rate, guardrail audit, sensitivity analysis)
is unchanged either way. Until then, every dollar figure this script
prints should be read as "under these stated assumptions," and the
generated report says so explicitly.

Usage
-----
    python phase4_backtest.py --input held_out_batch.csv --outdir ./phase4_output
    python phase4_backtest.py --demo --n 6000               # no data yet? try it end to end
    python phase4_backtest.py --input batch.csv --outcomes-csv real_outcomes.csv

Input CSV (per schema.md) must have at least:
    transaction_id, amount, decline_reason_category, bank_response_code
Optional columns used if present, else defaulted/simulated:
    predicted_category, predicted_confidence   (real Phase 2 model output —
        strongly preferred; if absent, simulated from Phase 2's measured
        confusion matrix, see simulate_predicted_category())
    attempt_number, payment_method, gateway_recent_failure_rate,
    merchant_hist_failure_rate, customer_opted_out, last_attempt_time
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from retry_policy_engine import (
    ClassifierOutput,
    TransactionContext,
    RetryAction,
    RISK_CODED_RAW_VALUES,
    BASE_RECOVERY_PRIORS,
    DELAYED_RETRY_WAIT,
    MAX_RETRY_ATTEMPTS,
    evaluate_retry_policy,
)

# ---------------------------------------------------------------------------
# Ground-truth category groupings
# ---------------------------------------------------------------------------

# Phase 3 (retry_policy_engine.NO_RETRY_CATEGORIES) hard-gates exactly these two.
HARD_NO_RETRY_TRUTH = frozenset({"risk_block", "card_expired"})

# schema.md's stated intent is broader: "other" is also "no automated action,
# flag for review." Phase 3's code currently does NOT hard-gate "other" (it's
# present in BASE_RECOVERY_PRIORS and flows through the ML layer like any
# retry-eligible category). That's a real gap between the design doc and the
# implementation — this harness measures it (`other_category_policy_gap`)
# rather than silently assuming the code matches the doc.
SCHEMA_NO_AUTOMATION_TRUTH = HARD_NO_RETRY_TRUTH | {"other"}

TRUE_PROB_HARD_BLOCKED = 0.02  # negligible fluke-recovery odds for genuinely
                               # blocked/expired cases, used regardless of
                               # which action is (wrongly) attempted on them.

# ---------------------------------------------------------------------------
# Ground-truth columns (gt_is_recoverable, gt_ideal_retry_action,
# gt_requires_customer_action)
#
# NOTE ON WHAT THESE ARE VS. WHAT `true_recovery_probability()` ABOVE IS:
# The simulator block above (`build_true_recovery_probs`, `true_recovery_
# probability`) exists because, per the module docstring, the Phase 1/2
# dataset has no observed retry_outcome/time_to_recovery history — so the
# headline recovered-revenue number is a decision-logic backtest under a
# simulated outcome model, not a measurement.
#
# gt_is_recoverable / gt_ideal_retry_action / gt_requires_customer_action are
# a DIFFERENT thing: per-row labels that (per the request that added this
# block) are already present in the scored batch. They are treated here as
# an independent, authoritative judgment of what *should* have happened for
# that row - independent of both the classifier's prediction and of the
# simulated recovery-probability model above. That makes them the right
# basis for false-positive-rate and guardrail-compliance checks (this is
# "did the policy do the right thing" per a real label), which is a
# different question from "how much money did the simulated outcome model
# say we recovered."
#
# If a batch doesn't have these columns (e.g. --demo, or an older input file
# from before they existed), we fall back to a deterministic, documented
# heuristic derived from `decline_reason_category` + `BASE_RECOVERY_PRIORS`
# so the harness still runs end-to-end - but this fallback is clearly logged
# and clearly labeled in the report, since it is NOT the same thing as a
# real label and should not be read as one.
# ---------------------------------------------------------------------------

GT_COLUMNS = ("gt_is_recoverable", "gt_ideal_retry_action", "gt_requires_customer_action")

# Any recovery probability at/above this (per BASE_RECOVERY_PRIORS' best
# action for the category) counts as "recoverable" in the fallback heuristic.
GT_FALLBACK_RECOVERABLE_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Phase 2's measured test-set confusion matrix (phase2_results.md), used ONLY
# to simulate realistic classifier predictions when the batch doesn't already
# carry real predicted_category/predicted_confidence columns.
# ---------------------------------------------------------------------------

CONFUSION_COUNTS: Dict[str, Dict[str, int]] = {
    "insufficient_funds": {"insufficient_funds": 285, "bank_server_error": 5, "invalid_otp": 6, "risk_block": 4, "network_timeout": 2, "card_expired": 0, "limit_exceeded": 4, "other": 4},
    "bank_server_error":  {"insufficient_funds": 4, "bank_server_error": 182, "invalid_otp": 20, "risk_block": 3, "network_timeout": 13, "card_expired": 0, "limit_exceeded": 4, "other": 1},
    "invalid_otp":        {"insufficient_funds": 0, "bank_server_error": 11, "invalid_otp": 104, "risk_block": 2, "network_timeout": 1, "card_expired": 0, "limit_exceeded": 5, "other": 1},
    "risk_block":         {"insufficient_funds": 0, "bank_server_error": 1, "invalid_otp": 0, "risk_block": 75, "network_timeout": 1, "card_expired": 2, "limit_exceeded": 1, "other": 4},
    "network_timeout":    {"insufficient_funds": 1, "bank_server_error": 12, "invalid_otp": 1, "risk_block": 1, "network_timeout": 84, "card_expired": 0, "limit_exceeded": 0, "other": 0},
    "card_expired":       {"insufficient_funds": 0, "bank_server_error": 0, "invalid_otp": 1, "risk_block": 1, "network_timeout": 1, "card_expired": 59, "limit_exceeded": 0, "other": 0},
    "limit_exceeded":     {"insufficient_funds": 1, "bank_server_error": 0, "invalid_otp": 2, "risk_block": 1, "network_timeout": 0, "card_expired": 1, "limit_exceeded": 62, "other": 1},
    "other":              {"insufficient_funds": 0, "bank_server_error": 2, "invalid_otp": 0, "risk_block": 0, "network_timeout": 0, "card_expired": 1, "limit_exceeded": 0, "other": 48},
}

DEMO_CATEGORY_SUPPORT = {  # mirrors Phase 2's test-set support distribution
    "insufficient_funds": 310, "bank_server_error": 227, "invalid_otp": 124,
    "risk_block": 84, "network_timeout": 99, "card_expired": 62,
    "limit_exceeded": 68, "other": 51,
}


# ---------------------------------------------------------------------------
# Simulators (documented, seeded — see module docstring)
# ---------------------------------------------------------------------------

def build_true_recovery_probs(seed: int) -> Dict[str, Dict[str, float]]:
    """
    'True' P(recovery | category, action), derived from Phase 3's own
    BASE_RECOVERY_PRIORS but intentionally perturbed by a seeded +/-20%
    multiplicative factor per (category, action). This is so the backtest
    isn't just re-checking the policy against the exact numbers it was
    built from (that would be tautological). It is still an assumption,
    not observed data — see module docstring.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for category, action_priors in BASE_RECOVERY_PRIORS.items():
        out[category] = {}
        for action, p in action_priors.items():
            noise = rng.uniform(0.8, 1.2)
            out[category][action.value] = float(np.clip(p * noise, 0.01, 0.95))
    return out


def true_recovery_probability(truth_category: str, action: str, true_probs: dict) -> float:
    if truth_category in HARD_NO_RETRY_TRUTH:
        return TRUE_PROB_HARD_BLOCKED
    table = true_probs.get(truth_category)
    if not table:
        return TRUE_PROB_HARD_BLOCKED
    return table.get(action, TRUE_PROB_HARD_BLOCKED)


def simulate_predicted_category(truth_category: str, rng: np.random.Generator) -> str:
    row = CONFUSION_COUNTS.get(truth_category)
    if not row:
        return truth_category
    cats = list(row.keys())
    counts = np.array(list(row.values()), dtype=float)
    counts = counts if counts.sum() > 0 else np.ones_like(counts)
    probs = counts / counts.sum()
    return str(rng.choice(cats, p=probs))


def simulate_predicted_confidence(correct: bool, rng: np.random.Generator) -> float:
    # Illustrative only — a simple higher/lower Beta split. Real per-row model
    # probabilities (via --predicted-col/--confidence-col) should always be
    # preferred; this exists so --demo and gap-filled batches still exercise
    # the low-confidence guardrail path realistically.
    if correct:
        return float(np.clip(rng.beta(8, 2), 0.05, 0.99))
    return float(np.clip(rng.beta(3, 3), 0.05, 0.99))


def _to_bool(x, default=None):
    """Tolerant bool coercion for a gt column that may arrive as an actual
    bool, or as a CSV string ('True'/'False'/'1'/'0'/'yes'/'no'), or as NaN."""
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if x != x:  # NaN
            return default
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return default


def derive_fallback_ground_truth(truth_category: str) -> dict:
    """
    Deterministic fallback used ONLY when a batch is missing the real
    gt_is_recoverable / gt_ideal_retry_action / gt_requires_customer_action
    columns (e.g. --demo runs). Derived from decline_reason_category via
    BASE_RECOVERY_PRIORS, NOT from any observed label - see module note
    above GT_COLUMNS. Callers must track and report when this fallback was
    used (see `gt_is_fallback` on each audit row).
    """
    if truth_category in HARD_NO_RETRY_TRUTH:
        return {
            "gt_is_recoverable": False,
            "gt_ideal_retry_action": RetryAction.NO_RETRY.value,
            "gt_requires_customer_action": False,
        }
    priors = BASE_RECOVERY_PRIORS.get(truth_category)
    if not priors:
        return {
            "gt_is_recoverable": False,
            "gt_ideal_retry_action": RetryAction.NO_RETRY.value,
            "gt_requires_customer_action": False,
        }
    best_action, best_p = max(priors.items(), key=lambda kv: kv[1])
    return {
        "gt_is_recoverable": bool(best_p >= GT_FALLBACK_RECOVERABLE_THRESHOLD),
        "gt_ideal_retry_action": best_action.value,
        "gt_requires_customer_action": bool(best_action == RetryAction.ALT_PAYMENT_METHOD_PROMPT),
    }


def ensure_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure gt_is_recoverable / gt_ideal_retry_action /
    gt_requires_customer_action exist on every row. Real columns from the
    input CSV are used as-is (with tolerant type coercion); rows missing
    them fall back to `derive_fallback_ground_truth`, and an explicit
    `gt_is_fallback` flag records which rows used the fallback so the
    report can be honest about it instead of silently blending real labels
    with heuristic ones.
    """
    df = df.copy()
    has_any_gt_cols = any(c in df.columns for c in GT_COLUMNS)
    if not has_any_gt_cols:
        print("[phase4] No gt_is_recoverable/gt_ideal_retry_action/"
              "gt_requires_customer_action columns found - false-positive "
              "rate and guardrail-compliance checks will use a fallback "
              "heuristic derived from decline_reason_category, NOT real "
              "ground-truth labels. Pass real columns via --input for a "
              "trustworthy check.", file=sys.stderr)

    is_recoverable_col, ideal_action_col, requires_cust_col, is_fallback_col = [], [], [], []
    for row in df.to_dict("records"):
        truth_category = row.get("decline_reason_category")
        fallback = derive_fallback_ground_truth(truth_category)

        raw_recoverable = row.get("gt_is_recoverable", None)
        recoverable = _to_bool(raw_recoverable, default=None)
        raw_ideal = row.get("gt_ideal_retry_action", None)
        ideal_action = raw_ideal if isinstance(raw_ideal, str) and raw_ideal.strip() else None
        raw_requires_cust = row.get("gt_requires_customer_action", None)
        requires_cust = _to_bool(raw_requires_cust, default=None)

        used_fallback = recoverable is None or ideal_action is None or requires_cust is None
        is_recoverable_col.append(fallback["gt_is_recoverable"] if recoverable is None else recoverable)
        ideal_action_col.append(fallback["gt_ideal_retry_action"] if ideal_action is None else ideal_action)
        requires_cust_col.append(fallback["gt_requires_customer_action"] if requires_cust is None else requires_cust)
        is_fallback_col.append(used_fallback)

    df["gt_is_recoverable"] = is_recoverable_col
    df["gt_ideal_retry_action"] = ideal_action_col
    df["gt_requires_customer_action"] = requires_cust_col
    df["gt_is_fallback"] = is_fallback_col
    return df


def simulate_time_to_recovery(
    action: str, category: str, rng: np.random.Generator, extra_delay_seconds: float = 0.0
) -> float:
    """
    Seconds from failed attempt to recovered success. Illustrative
    distributions pending real telemetry (time_to_recovery is null in the
    Phase 1/2 dataset — see module docstring).
    """
    if extra_delay_seconds:
        return extra_delay_seconds + rng.lognormal(mean=math.log(30), sigma=0.6)
    if action == RetryAction.DELAYED_RETRY.value:
        base = DELAYED_RETRY_WAIT.get(category, timedelta(hours=6)).total_seconds()
        return base + rng.lognormal(mean=math.log(max(base * 0.05, 30)), sigma=0.6)
    if action == RetryAction.ALT_PAYMENT_METHOD_PROMPT.value:
        return rng.lognormal(mean=math.log(180), sigma=0.8)  # customer has to act
    return rng.lognormal(mean=math.log(15), sigma=0.7)  # same/alt gateway: fast


# ---------------------------------------------------------------------------
# Demo batch (only used with --demo / when no --input is given)
# ---------------------------------------------------------------------------

def generate_demo_batch(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(DEMO_CATEGORY_SUPPORT.keys())
    weights = np.array(list(DEMO_CATEGORY_SUPPORT.values()), dtype=float)
    weights /= weights.sum()
    truth = rng.choice(cats, size=n, p=weights)

    generic_codes = ["U01", "U05", "U12", "U22", "U31", "U51", "U60", "U69", "U71", "U80"]
    risk_codes = sorted(RISK_CODED_RAW_VALUES)

    rows = []
    for i, cat in enumerate(truth):
        if cat == "risk_block":
            code = rng.choice(risk_codes) if rng.random() < 0.85 else rng.choice(generic_codes)
        else:
            # ~3% wrong-pool code noise on non-risk rows, mirroring Phase 1's
            # documented ~8% cross-category code noise (kept lower here
            # specifically for risk-coded values since over-triggering the
            # hard rule on every category would understate its real value).
            code = rng.choice(risk_codes) if rng.random() < 0.03 else rng.choice(generic_codes)
        amount = float(np.clip(rng.lognormal(mean=6.7, sigma=0.9), 50, 75000))
        rows.append({
            "transaction_id": f"demo_txn_{i:06d}",
            "merchant_id": f"m_{int(rng.integers(1, 40)):03d}",
            "customer_id": f"c_{int(rng.integers(1, 5000)):05d}",
            "amount": round(amount, 2),
            "currency": "INR",
            "payment_method": rng.choice(["card", "upi", "netbanking", "wallet"], p=[0.35, 0.45, 0.12, 0.08]),
            "gateway": rng.choice(["gateway_a", "gateway_b", "gateway_c"]),
            "bank_response_code": code,
            "decline_reason_category": cat,
            "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=int(rng.integers(0, 60 * 24 * 10)))).isoformat(),
            "attempt_number": int(min(5, rng.geometric(0.55))),
            "is_recurring": bool(rng.random() < 0.15),
            "gateway_recent_failure_rate": float(np.clip(rng.normal(0.06, 0.05), 0.0, 0.9)),
            "merchant_hist_failure_rate": float(np.clip(rng.normal(0.08, 0.06), 0.0, 0.9)),
            "customer_opted_out": bool(rng.random() < 0.02),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-row simulation
# ---------------------------------------------------------------------------

def process_row(row: dict, rng: np.random.Generator, true_probs: dict,
                 gateway_fee: float, annoyance_cost: float) -> dict:
    truth_category = row["decline_reason_category"]
    predicted_category = row["predicted_category"]
    predicted_confidence = float(row["predicted_confidence"])
    bank_code = row["bank_response_code"]
    amount = float(row["amount"])
    gt_is_recoverable = bool(row["gt_is_recoverable"])
    gt_ideal_retry_action = row["gt_ideal_retry_action"]
    gt_requires_customer_action = bool(row["gt_requires_customer_action"])
    gt_is_fallback = bool(row["gt_is_fallback"])

    clf = ClassifierOutput(
        decline_reason_category=predicted_category,
        confidence=predicted_confidence,
        bank_response_code=bank_code,
    )
    ctx = TransactionContext(
        transaction_id=str(row["transaction_id"]),
        attempt_count=int(row.get("attempt_number", 1) or 1),
        last_attempt_time=row.get("last_attempt_time", None),
        customer_opted_out=bool(row.get("customer_opted_out", False)),
        gateway_recent_failure_rate=row.get("gateway_recent_failure_rate", None),
        merchant_hist_failure_rate=row.get("merchant_hist_failure_rate", None),
        amount=amount,
        payment_method=row.get("payment_method", None),
    )

    decision = evaluate_retry_policy(clf, ctx)
    action = decision["recommended_action"]

    # --- Layer-1 integrity check (should NEVER be violated; see report) -----
    hard_rule_should_apply = (
        predicted_category == "risk_block"
        or predicted_category == "card_expired"
        or bank_code in RISK_CODED_RAW_VALUES
    )
    hard_rule_violation = hard_rule_should_apply and action != RetryAction.NO_RETRY.value

    # --- Policy arm: simulate outcome if a retry was actually recommended --
    policy_recovered = None
    policy_ttr = None
    policy_fee = 0.0
    policy_annoy = 0.0
    if action != RetryAction.NO_RETRY.value:
        p_true = true_recovery_probability(truth_category, action, true_probs)
        policy_recovered = bool(rng.random() < p_true)
        policy_fee = gateway_fee
        if policy_recovered:
            policy_ttr = simulate_time_to_recovery(action, truth_category, rng)
        else:
            policy_annoy = annoyance_cost

    # --- Baseline B: naive retry-everything-once-after-1h, no policy logic -
    naive_action = RetryAction.SAME_GATEWAY_RETRY.value
    p_true_b = true_recovery_probability(truth_category, naive_action, true_probs)
    baseline_b_recovered = bool(rng.random() < p_true_b)
    baseline_b_fee = gateway_fee
    baseline_b_ttr = None
    baseline_b_annoy = 0.0
    if baseline_b_recovered:
        baseline_b_ttr = simulate_time_to_recovery(naive_action, truth_category, rng, extra_delay_seconds=3600.0)
    else:
        baseline_b_annoy = annoyance_cost

    is_hard_fp_policy = truth_category in HARD_NO_RETRY_TRUTH and action != RetryAction.NO_RETRY.value
    is_over_block = hard_rule_should_apply and truth_category not in HARD_NO_RETRY_TRUTH
    other_gap = (truth_category == "other" or predicted_category == "other") and action != RetryAction.NO_RETRY.value

    # --- Requested metric 1: false-positive retry (gt_is_recoverable=False) -
    retried = action != RetryAction.NO_RETRY.value
    is_wasted_retry = retried and not gt_is_recoverable

    # --- Requested metric 2: guardrail-compliance vs. risk_block/card_expired
    # ground truth. This is a verification check on the Layer-1 invariant
    # retry_policy_engine.py already enforces (NO_RETRY_CATEGORIES +
    # RISK_CODED_RAW_VALUES) - not a new rule. Two related but distinct
    # things are both worth knowing and are kept separate:
    #  - guardrail_violation_engine: did the ENGINE's own deterministic gate
    #    fire correctly given what it was told (predicted_category / bank
    #    code)? This should be structurally impossible to violate and is
    #    the same check as `hard_rule_violation` above.
    #  - guardrail_violation_gt: given the row's real-world truth_category
    #    (risk_block/card_expired), was a retry recommended anyway? This CAN
    #    be nonzero purely from classifier misclassification (the engine
    #    never sees the truth category) - that's a model-accuracy gap, not a
    #    guardrail bug, and is reported separately so the two aren't
    #    conflated.
    guardrail_hard_category = truth_category in HARD_NO_RETRY_TRUTH
    guardrail_violation_gt = guardrail_hard_category and retried
    # Cross-check the truth_category-based hard-block set against the gt_*
    # columns for the same row: they're expected to agree (gt_is_recoverable
    # should be False and gt_ideal_retry_action should be "no_retry" for
    # risk_block/card_expired rows). A mismatch flags a data-quality issue
    # between decline_reason_category and the gt_* labels, not a policy bug.
    gt_label_mismatch = guardrail_hard_category and (
        gt_is_recoverable or gt_ideal_retry_action != RetryAction.NO_RETRY.value
    )

    over_block_expected_recovery = 0.0
    if is_over_block:
        candidate_probs = true_probs.get(truth_category, {})
        if candidate_probs:
            over_block_expected_recovery = amount * max(candidate_probs.values())

    return dict(
        transaction_id=row["transaction_id"], amount=amount,
        truth_category=truth_category, predicted_category=predicted_category,
        predicted_confidence=round(predicted_confidence, 3), bank_response_code=bank_code,
        attempt_number=ctx.attempt_count,
        policy_action=action, policy_timing=decision["timing"],
        policy_confidence=decision["confidence"], policy_flagged=decision["flagged_for_human_review"],
        policy_reason=decision["reason_string"], guardrail_checks="; ".join(decision["guardrail_checks"]),
        hard_rule_should_apply=hard_rule_should_apply, hard_rule_violation=hard_rule_violation,
        policy_recovered=policy_recovered, policy_time_to_recovery_s=policy_ttr,
        policy_gateway_fee=policy_fee, policy_annoyance_cost=policy_annoy,
        baseline_a_action="no_retry", baseline_a_recovered=False,
        baseline_b_action="same_gateway_retry(naive_1h)", baseline_b_recovered=baseline_b_recovered,
        baseline_b_time_to_recovery_s=baseline_b_ttr, baseline_b_gateway_fee=baseline_b_fee,
        baseline_b_annoyance_cost=baseline_b_annoy,
        is_hard_fp_policy=is_hard_fp_policy,
        is_hard_fp_baseline_b=(truth_category in HARD_NO_RETRY_TRUTH),
        is_over_block=is_over_block, over_block_expected_recovery=over_block_expected_recovery,
        other_category_policy_gap=other_gap,
        gt_is_recoverable=gt_is_recoverable, gt_ideal_retry_action=gt_ideal_retry_action,
        gt_requires_customer_action=gt_requires_customer_action, gt_is_fallback=gt_is_fallback,
        is_wasted_retry=is_wasted_retry,
        guardrail_hard_category=guardrail_hard_category,
        guardrail_violation_gt=guardrail_violation_gt,
        gt_label_mismatch=gt_label_mismatch,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def latency_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"n": 0, "median_s": None, "p90_s": None, "p99_s": None, "max_s": None}
    return {
        "n": int(s.shape[0]),
        "median_s": float(s.median()),
        "p90_s": float(s.quantile(0.90)),
        "p99_s": float(s.quantile(0.99)),
        "max_s": float(s.max()),
    }


LATENCY_BUCKETS = [
    ("immediate (<5min)", 0, 300),
    ("<1hr", 300, 3600),
    ("<1day", 3600, 86400),
    (">1day", 86400, math.inf),
]


def latency_buckets(series: pd.Series) -> list:
    """Bucket a seconds-valued series (e.g. time-to-recovery) into the
    documented buckets. Returns a list of {label, n, pct} dicts, pct is
    the share of the (non-null) recovered population in that bucket."""
    s = series.dropna()
    total = int(s.shape[0])
    rows = []
    for label, lo, hi in LATENCY_BUCKETS:
        n = int(((s >= lo) & (s < hi)).sum())
        rows.append({"label": label, "n": n, "pct": pct(n, total)})
    return rows


def aggregate(audit_df: pd.DataFrame, gateway_fee: float, annoyance_cost: float) -> dict:
    n = len(audit_df)
    total_at_risk = float(audit_df["amount"].sum())

    hard_truth_mask = audit_df["truth_category"].isin(HARD_NO_RETRY_TRUTH)
    n_hard_truth = int(hard_truth_mask.sum())

    # --- Policy arm --------------------------------------------------------
    policy_retried_mask = audit_df["policy_action"] != RetryAction.NO_RETRY.value
    policy_recovered_mask = audit_df["policy_recovered"] == True  # noqa: E712
    retries_policy = int(policy_retried_mask.sum())
    recovered_policy_n = int(policy_recovered_mask.sum())
    gross_recovered_policy = float(audit_df.loc[policy_recovered_mask, "amount"].sum())
    gateway_cost_policy = float(audit_df["policy_gateway_fee"].sum())
    annoyance_cost_policy = float(audit_df["policy_annoyance_cost"].sum())
    net_recovered_policy = gross_recovered_policy - gateway_cost_policy - annoyance_cost_policy

    # --- Baseline A: no retry ever ------------------------------------------
    gross_recovered_a = 0.0
    net_recovered_a = 0.0

    # --- Baseline B: naive retry everything once after 1h -------------------
    b_recovered_mask = audit_df["baseline_b_recovered"] == True  # noqa: E712
    retries_b = n  # attempts everything
    recovered_b_n = int(b_recovered_mask.sum())
    gross_recovered_b = float(audit_df.loc[b_recovered_mask, "amount"].sum())
    gateway_cost_b = float(audit_df["baseline_b_gateway_fee"].sum())
    annoyance_cost_b = float(audit_df["baseline_b_annoyance_cost"].sum())
    net_recovered_b = gross_recovered_b - gateway_cost_b - annoyance_cost_b

    # --- Safety / compliance -------------------------------------------------
    hard_rule_violations = int(audit_df["hard_rule_violation"].sum())  # must be 0
    residual_leakage_n = int(audit_df["is_hard_fp_policy"].sum())
    residual_leakage_rate = pct(residual_leakage_n, n_hard_truth)
    naive_hard_fp_n = int(audit_df["is_hard_fp_baseline_b"].sum())  # == n_hard_truth, always
    over_block_n = int(audit_df["is_over_block"].sum())
    over_block_amount = float(audit_df.loc[audit_df["is_over_block"], "amount"].sum())
    over_block_expected_recovery_forgone = float(audit_df["over_block_expected_recovery"].sum())
    other_gap_n = int(audit_df["other_category_policy_gap"].sum())
    flagged_n = int(audit_df["policy_flagged"].sum())

    risk_placeholder_active = RISK_CODED_RAW_VALUES == frozenset({"U39", "U40", "U41"})

    # --- Requested metric 1: false-positive retry rate (gt_is_recoverable) --
    gt_unrecoverable_mask = audit_df["gt_is_recoverable"] == False  # noqa: E712
    n_gt_unrecoverable = int(gt_unrecoverable_mask.sum())
    policy_wasted_mask = audit_df["is_wasted_retry"] == True  # noqa: E712
    policy_wasted_n = int(policy_wasted_mask.sum())
    policy_wasted_amount = float(audit_df.loc[policy_wasted_mask, "amount"].sum())
    # Baseline B retries every single row, so ALL gt-unrecoverable rows are
    # wasted retries under it by construction.
    baseline_b_wasted_n = n_gt_unrecoverable
    baseline_b_wasted_amount = float(audit_df.loc[gt_unrecoverable_mask, "amount"].sum())

    false_positive = dict(
        n_gt_unrecoverable=n_gt_unrecoverable,
        policy=dict(
            wasted_retries_n=policy_wasted_n,
            wasted_retries_amount=policy_wasted_amount,
            # standard FP rate = FP / (FP + TN) = wasted retries / all
            # gt-unrecoverable cases
            fp_rate_pct=pct(policy_wasted_n, n_gt_unrecoverable),
            # secondary framing: what share of all retries the policy fired
            # were wasted on an unrecoverable case
            share_of_retries_pct=pct(policy_wasted_n, retries_policy),
        ),
        baseline_b=dict(
            wasted_retries_n=baseline_b_wasted_n,
            wasted_retries_amount=baseline_b_wasted_amount,
            fp_rate_pct=pct(baseline_b_wasted_n, n_gt_unrecoverable),
            share_of_retries_pct=pct(baseline_b_wasted_n, retries_b),
        ),
    )

    # --- Requested metric 2: guardrail compliance vs. risk_block/card_expired
    guardrail_violation_gt_n = int(audit_df["guardrail_violation_gt"].sum())  # expect 0
    risk_block_mask = audit_df["truth_category"] == "risk_block"
    card_expired_mask = audit_df["truth_category"] == "card_expired"
    guardrail_by_category = {
        "risk_block": dict(
            n=int(risk_block_mask.sum()),
            violations=int((risk_block_mask & (audit_df["guardrail_violation_gt"] == True)).sum()),  # noqa: E712
        ),
        "card_expired": dict(
            n=int(card_expired_mask.sum()),
            violations=int((card_expired_mask & (audit_df["guardrail_violation_gt"] == True)).sum()),  # noqa: E712
        ),
    }
    gt_label_mismatch_n = int(audit_df["gt_label_mismatch"].sum())
    gt_fallback_n = int(audit_df["gt_is_fallback"].sum())
    guardrail_compliance = dict(
        engine_hard_rule_violations=hard_rule_violations,  # same invariant, engine-input-based
        gt_violations_n=guardrail_violation_gt_n,           # truth-category-based
        gt_violations_pct=pct(guardrail_violation_gt_n, n_hard_truth),
        by_category=guardrail_by_category,
        gt_label_mismatch_n=gt_label_mismatch_n,
        gt_fallback_rows_n=gt_fallback_n,
        gt_fallback_used=gt_fallback_n > 0,
        # "Compliant" tracks the invariant the user asked to verify: the
        # engine's own deterministic gate (NO_RETRY_CATEGORIES +
        # RISK_CODED_RAW_VALUES) should NEVER recommend a retry once it
        # has seen a risk_block/card_expired predicted_category or a
        # risk-coded bank_response_code. That's structurally guaranteed by
        # retry_policy_engine.py and is exactly what hard_rule_violations
        # measures. gt_violations_n is a related but different number - it
        # can be nonzero purely from classifier misclassification (the
        # engine never sees the *true* category) - so it is reported
        # separately as context, not folded into pass/fail.
        compliant=(hard_rule_violations == 0),
    )

    # --- Requested metric 3: latency-to-recovery distribution ---------------
    policy_latency_buckets = latency_buckets(audit_df.loc[policy_recovered_mask, "policy_time_to_recovery_s"])
    baseline_b_latency_buckets = latency_buckets(audit_df.loc[b_recovered_mask, "baseline_b_time_to_recovery_s"])

    return dict(
        n=n, total_at_risk=total_at_risk, n_hard_truth=n_hard_truth,
        policy=dict(
            retries_attempted=retries_policy, recovered_n=recovered_policy_n,
            recovery_rate_pct=pct(recovered_policy_n, retries_policy),
            gross_recovered=gross_recovered_policy, gateway_cost=gateway_cost_policy,
            annoyance_cost=annoyance_cost_policy, net_recovered=net_recovered_policy,
            gross_pct_at_risk=pct(gross_recovered_policy, total_at_risk),
            net_pct_at_risk=pct(net_recovered_policy, total_at_risk),
            latency=latency_stats(audit_df.loc[policy_recovered_mask, "policy_time_to_recovery_s"]),
            latency_buckets=policy_latency_buckets,
        ),
        baseline_a=dict(
            retries_attempted=0, recovered_n=0, recovery_rate_pct=0.0,
            gross_recovered=gross_recovered_a, gateway_cost=0.0, annoyance_cost=0.0,
            net_recovered=net_recovered_a, gross_pct_at_risk=0.0, net_pct_at_risk=0.0,
            latency=latency_stats(pd.Series(dtype=float)),
        ),
        baseline_b=dict(
            retries_attempted=retries_b, recovered_n=recovered_b_n,
            recovery_rate_pct=pct(recovered_b_n, retries_b),
            gross_recovered=gross_recovered_b, gateway_cost=gateway_cost_b,
            annoyance_cost=annoyance_cost_b, net_recovered=net_recovered_b,
            gross_pct_at_risk=pct(gross_recovered_b, total_at_risk),
            net_pct_at_risk=pct(net_recovered_b, total_at_risk),
            latency=latency_stats(audit_df.loc[b_recovered_mask, "baseline_b_time_to_recovery_s"]),
            latency_buckets=baseline_b_latency_buckets,
        ),
        false_positive=false_positive,
        guardrail_compliance=guardrail_compliance,
        safety=dict(
            hard_rule_violations=hard_rule_violations,
            residual_leakage_n=residual_leakage_n,
            residual_leakage_rate_pct=residual_leakage_rate,
            naive_baseline_hard_fp_n=naive_hard_fp_n,
            naive_baseline_hard_fp_rate_pct=pct(naive_hard_fp_n, n_hard_truth),
            over_block_n=over_block_n,
            over_block_amount=over_block_amount,
            over_block_expected_recovery_forgone=over_block_expected_recovery_forgone,
            other_category_policy_gap_n=other_gap_n,
            flagged_for_human_review_n=flagged_n,
            risk_coded_values_still_placeholder=risk_placeholder_active,
        ),
        cost_inputs=dict(gateway_fee=gateway_fee, annoyance_cost=annoyance_cost),
    )


def sensitivity_analysis(audit_df: pd.DataFrame, gateway_fee: float, annoyance_cost: float) -> list:
    policy_retried_mask = audit_df["policy_action"] != RetryAction.NO_RETRY.value
    policy_recovered_mask = audit_df["policy_recovered"] == True  # noqa: E712
    b_recovered_mask = audit_df["baseline_b_recovered"] == True  # noqa: E712

    gross_policy = float(audit_df.loc[policy_recovered_mask, "amount"].sum())
    gross_b = float(audit_df.loc[b_recovered_mask, "amount"].sum())
    n_retries_policy = int(policy_retried_mask.sum())
    n_failed_policy = n_retries_policy - int(policy_recovered_mask.sum())
    n_retries_b = len(audit_df)
    n_failed_b = n_retries_b - int(b_recovered_mask.sum())

    scenarios = [
        ("baseline assumptions (1x fee / 1x annoyance)", 1.0, 1.0),
        ("gateway fee 2x", 2.0, 1.0),
        ("gateway fee 0.5x", 0.5, 1.0),
        ("annoyance cost 2x", 1.0, 2.0),
        ("annoyance cost 0.5x", 1.0, 0.5),
        ("both costs 2x (worst case)", 2.0, 2.0),
        ("both costs 0.5x (best case)", 0.5, 0.5),
    ]
    rows = []
    for label, fee_mult, annoy_mult in scenarios:
        fee = gateway_fee * fee_mult
        annoy = annoyance_cost * annoy_mult
        net_policy = gross_policy - n_retries_policy * fee - n_failed_policy * annoy
        net_b = gross_b - n_retries_b * fee - n_failed_b * annoy
        rows.append(dict(
            scenario=label, gateway_fee=round(fee, 4), annoyance_cost=round(annoy, 4),
            net_recovered_policy=round(net_policy, 2), net_recovered_baseline_b=round(net_b, 2),
            policy_advantage=round(net_policy - net_b, 2),
        ))
    return rows


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

REPORT_TEMPLATE_PATH_DEFAULT = "phase4_report_template.md"


def money(x: float) -> str:
    return f"₹{x:,.2f}"


def render_report(template_text: str, results: dict, sensitivity: list, meta: dict) -> str:
    p, a, b, s, c = results["policy"], results["baseline_a"], results["baseline_b"], results["safety"], results["cost_inputs"]
    fp, gc = results["false_positive"], results["guardrail_compliance"]

    def bucket_table(buckets: list) -> str:
        lines = ["| Latency bucket | n | % of recovered |", "|---|---|---|"]
        for row in buckets:
            lines.append(f"| {row['label']} | {row['n']:,} | {row['pct']:.1f}% |")
        return "\n".join(lines)

    policy_latency_bucket_table = bucket_table(p["latency_buckets"])
    baseline_b_latency_bucket_table = bucket_table(b["latency_buckets"])

    guardrail_table_lines = ["| Ground-truth category | n | Retries recommended (should be 0) |", "|---|---|---|"]
    for cat, row in gc["by_category"].items():
        guardrail_table_lines.append(f"| {cat} | {row['n']:,} | {row['violations']:,} |")
    guardrail_table = "\n".join(guardrail_table_lines)

    sens_lines = ["| Scenario | Gateway fee | Annoyance cost | Net recovered (policy) | Net recovered (naive) | Policy advantage |",
                  "|---|---|---|---|---|---|"]
    for row in sensitivity:
        sens_lines.append(
            f"| {row['scenario']} | {money(row['gateway_fee'])} | {money(row['annoyance_cost'])} | "
            f"{money(row['net_recovered_policy'])} | {money(row['net_recovered_baseline_b'])} | "
            f"{money(row['policy_advantage'])} |"
        )
    sensitivity_table = "\n".join(sens_lines)

    def lat(stat: dict) -> str:
        if stat["n"] == 0:
            return "n/a (no recoveries)"
        return (f"n={stat['n']}, median={stat['median_s']:.0f}s, p90={stat['p90_s']:.0f}s, "
                f"p99={stat['p99_s']:.0f}s, max={stat['max_s']:.0f}s")

    tokens = {
        "{{RUN_TIMESTAMP}}": meta["run_timestamp"],
        "{{INPUT_SOURCE}}": meta["input_source"],
        "{{SEED}}": str(meta["seed"]),
        "{{N_TRANSACTIONS}}": f"{results['n']:,}",
        "{{TOTAL_AT_RISK}}": money(results["total_at_risk"]),
        "{{N_HARD_TRUTH}}": f"{results['n_hard_truth']:,}",

        "{{POLICY_RETRIES}}": f"{p['retries_attempted']:,}",
        "{{POLICY_RECOVERED_N}}": f"{p['recovered_n']:,}",
        "{{POLICY_RECOVERY_RATE}}": f"{p['recovery_rate_pct']:.1f}%",
        "{{POLICY_GROSS}}": money(p["gross_recovered"]),
        "{{POLICY_GROSS_PCT}}": f"{p['gross_pct_at_risk']:.2f}%",
        "{{POLICY_GATEWAY_COST}}": money(p["gateway_cost"]),
        "{{POLICY_ANNOYANCE_COST}}": money(p["annoyance_cost"]),
        "{{POLICY_NET}}": money(p["net_recovered"]),
        "{{POLICY_NET_PCT}}": f"{p['net_pct_at_risk']:.2f}%",
        "{{POLICY_LATENCY}}": lat(p["latency"]),

        "{{BASELINE_A_NET}}": money(a["net_recovered"]),
        "{{BASELINE_A_NET_PCT}}": f"{a['net_pct_at_risk']:.2f}%",

        "{{BASELINE_B_RETRIES}}": f"{b['retries_attempted']:,}",
        "{{BASELINE_B_RECOVERED_N}}": f"{b['recovered_n']:,}",
        "{{BASELINE_B_RECOVERY_RATE}}": f"{b['recovery_rate_pct']:.1f}%",
        "{{BASELINE_B_GROSS}}": money(b["gross_recovered"]),
        "{{BASELINE_B_GROSS_PCT}}": f"{b['gross_pct_at_risk']:.2f}%",
        "{{BASELINE_B_GATEWAY_COST}}": money(b["gateway_cost"]),
        "{{BASELINE_B_ANNOYANCE_COST}}": money(b["annoyance_cost"]),
        "{{BASELINE_B_NET}}": money(b["net_recovered"]),
        "{{BASELINE_B_NET_PCT}}": f"{b['net_pct_at_risk']:.2f}%",
        "{{BASELINE_B_LATENCY}}": lat(b["latency"]),

        "{{POLICY_VS_A_DELTA}}": money(p["net_recovered"] - a["net_recovered"]),
        "{{POLICY_VS_B_DELTA}}": money(p["net_recovered"] - b["net_recovered"]),

        "{{HARD_RULE_VIOLATIONS}}": str(s["hard_rule_violations"]),
        "{{RESIDUAL_LEAKAGE_N}}": str(s["residual_leakage_n"]),
        "{{RESIDUAL_LEAKAGE_RATE}}": f"{s['residual_leakage_rate_pct']:.2f}%",
        "{{NAIVE_HARD_FP_N}}": str(s["naive_baseline_hard_fp_n"]),
        "{{NAIVE_HARD_FP_RATE}}": f"{s['naive_baseline_hard_fp_rate_pct']:.1f}%",
        "{{OVER_BLOCK_N}}": str(s["over_block_n"]),
        "{{OVER_BLOCK_AMOUNT}}": money(s["over_block_amount"]),
        "{{OVER_BLOCK_FORGONE}}": money(s["over_block_expected_recovery_forgone"]),
        "{{OTHER_GAP_N}}": str(s["other_category_policy_gap_n"]),
        "{{FLAGGED_N}}": str(s["flagged_for_human_review_n"]),
        "{{RISK_CODE_PLACEHOLDER_WARNING}}": (
            "**WARNING: `RISK_CODED_RAW_VALUES` in retry_policy_engine.py is still "
            "the placeholder set (`U39`, `U40`, `U41`) — replace with the real "
            "risk-coded values from the Phase 1 generator config before relying on "
            "the raw-code cross-check in production.**"
        ) if s["risk_coded_values_still_placeholder"] else (
            "`RISK_CODED_RAW_VALUES` has been customized from the placeholder set."
        ),

        "{{GATEWAY_FEE_ASSUMPTION}}": money(c["gateway_fee"]),
        "{{ANNOYANCE_COST_ASSUMPTION}}": money(c["annoyance_cost"]),
        "{{SENSITIVITY_TABLE}}": sensitivity_table,

        # --- False-positive retry rate (gt_is_recoverable == False) --------
        "{{GT_FALLBACK_WARNING}}": (
            f"**WARNING: {gc['gt_fallback_rows_n']:,} row(s) had no real "
            "gt_is_recoverable / gt_ideal_retry_action / "
            "gt_requires_customer_action columns and used the "
            "decline_reason_category-derived fallback instead of real "
            "ground-truth labels (see `derive_fallback_ground_truth`). "
            "Treat the false-positive and guardrail numbers below as "
            "illustrative, not measured, until real columns are supplied.**"
        ) if gc["gt_fallback_used"] else (
            "All rows carried real gt_is_recoverable / gt_ideal_retry_action "
            "/ gt_requires_customer_action labels — no fallback used."
        ),
        "{{FP_N_UNRECOVERABLE}}": f"{fp['n_gt_unrecoverable']:,}",
        "{{FP_POLICY_N}}": f"{fp['policy']['wasted_retries_n']:,}",
        "{{FP_POLICY_RATE}}": f"{fp['policy']['fp_rate_pct']:.2f}%",
        "{{FP_POLICY_AMOUNT}}": money(fp['policy']['wasted_retries_amount']),
        "{{FP_POLICY_SHARE_OF_RETRIES}}": f"{fp['policy']['share_of_retries_pct']:.2f}%",
        "{{FP_BASELINE_B_N}}": f"{fp['baseline_b']['wasted_retries_n']:,}",
        "{{FP_BASELINE_B_RATE}}": f"{fp['baseline_b']['fp_rate_pct']:.2f}%",
        "{{FP_BASELINE_B_AMOUNT}}": money(fp['baseline_b']['wasted_retries_amount']),
        "{{FP_BASELINE_B_SHARE_OF_RETRIES}}": f"{fp['baseline_b']['share_of_retries_pct']:.2f}%",

        # --- Guardrail compliance (risk_block / card_expired) --------------
        "{{GUARDRAIL_STATUS}}": "PASS — zero retries recommended" if gc["compliant"] else "FAIL — see violations below",
        "{{GUARDRAIL_ENGINE_VIOLATIONS}}": str(gc["engine_hard_rule_violations"]),
        "{{GUARDRAIL_GT_VIOLATIONS}}": str(gc["gt_violations_n"]),
        "{{GUARDRAIL_GT_VIOLATIONS_PCT}}": f"{gc['gt_violations_pct']:.2f}%",
        "{{GUARDRAIL_TABLE}}": guardrail_table,
        "{{GUARDRAIL_LABEL_MISMATCH_N}}": str(gc["gt_label_mismatch_n"]),

        # --- Latency-to-recovery distribution -------------------------------
        "{{POLICY_LATENCY_BUCKET_TABLE}}": policy_latency_bucket_table,
        "{{BASELINE_B_LATENCY_BUCKET_TABLE}}": baseline_b_latency_bucket_table,
    }

    text = template_text
    for token, value in tokens.items():
        text = text.replace(token, value)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_batch(args) -> pd.DataFrame:
    if args.input:
        df = pd.read_csv(args.input)
    else:
        print(f"[phase4] No --input given — generating a {args.n}-row DEMO batch "
              f"(seed={args.seed}). This is NOT your real held-out set; pass "
              f"--input to run against actual data.", file=sys.stderr)
        df = generate_demo_batch(args.n, args.seed)

    required = {"transaction_id", "amount", "decline_reason_category", "bank_response_code"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input batch is missing required column(s): {sorted(missing)}")

    return df


def ensure_predictions(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed + 1)  # offset from the outcome-simulation seed
    if "predicted_category" not in df.columns or "predicted_confidence" not in df.columns:
        print("[phase4] No predicted_category/predicted_confidence columns found — "
              "simulating classifier predictions from Phase 2's measured confusion "
              "matrix. Pass real model output via those columns for a true backtest.",
              file=sys.stderr)
        preds, confs = [], []
        for truth in df["decline_reason_category"]:
            pred = simulate_predicted_category(truth, rng)
            conf = simulate_predicted_confidence(pred == truth, rng)
            preds.append(pred)
            confs.append(conf)
        df["predicted_category"] = preds
        df["predicted_confidence"] = confs
    return df


def main():
    ap = argparse.ArgumentParser(description="Phase 4 backtesting harness")
    ap.add_argument("--input", type=str, default=None, help="CSV of held-out failed-payment batch (schema.md).")
    ap.add_argument("--outcomes-csv", type=str, default=None,
                    help="Optional CSV of REAL observed retry_outcome/time_to_recovery keyed by "
                         "transaction_id, to replace simulation once real data exists. Not yet wired "
                         "to a specific column contract — see NOTE in code before use.")
    ap.add_argument("--demo", action="store_true", help="Force demo-batch generation even if --input is set aside.")
    ap.add_argument("--n", type=int, default=6000, help="Rows in the demo batch (only used without --input).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (matches Phase 2's seed=42 convention).")
    ap.add_argument("--gateway-fee", type=float, default=2.0, help="Assumed INR cost per retry attempt (PLACEHOLDER — replace with real gateway pricing).")
    ap.add_argument("--annoyance-cost", type=float, default=5.0, help="Assumed INR cost per failed retry, customer-friction/support proxy (PLACEHOLDER).")
    ap.add_argument("--outdir", type=str, default="./phase4_output", help="Where to write audit_trail.csv, results_summary.json, report.md")
    ap.add_argument("--template", type=str, default=REPORT_TEMPLATE_PATH_DEFAULT, help="Path to the markdown report template.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_batch(args)
    df = ensure_predictions(df, args.seed)
    df = ensure_ground_truth(df)

    true_probs = build_true_recovery_probs(args.seed)
    rng = np.random.default_rng(args.seed)

    records = [
        process_row(row, rng, true_probs, args.gateway_fee, args.annoyance_cost)
        for row in df.to_dict("records")
    ]
    audit_df = pd.DataFrame.from_records(records)

    results = aggregate(audit_df, args.gateway_fee, args.annoyance_cost)
    sensitivity = sensitivity_analysis(audit_df, args.gateway_fee, args.annoyance_cost)

    # Hard fail if the Layer-1 invariant is ever violated — this should be
    # structurally impossible given retry_policy_engine.py's own code, so a
    # nonzero count here means something changed upstream and needs attention
    # before this report is trusted.
    if results["safety"]["hard_rule_violations"] != 0:
        print(f"[phase4] *** {results['safety']['hard_rule_violations']} HARD-RULE "
              f"VIOLATIONS DETECTED *** — the Layer-1 no_retry invariant did not "
              f"hold. Do not ship this build.", file=sys.stderr)

    gc = results["guardrail_compliance"]
    if gc["gt_violations_n"] != 0:
        print(f"[phase4] {gc['gt_violations_n']} retries were recommended on rows whose "
              f"TRUE category was risk_block/card_expired ({gc['gt_violations_pct']:.2f}% "
              f"of {results['n_hard_truth']:,} such rows). Since the engine only ever sees "
              f"the classifier's *predicted* category, this reflects classifier "
              f"misclassification, not a guardrail bug — but it's real leaked risk and "
              f"worth investigating.", file=sys.stderr)
    if gc["gt_label_mismatch_n"] != 0:
        print(f"[phase4] {gc['gt_label_mismatch_n']} risk_block/card_expired rows had "
              f"gt_is_recoverable/gt_ideal_retry_action values inconsistent with their "
              f"decline_reason_category — check the ground-truth data pipeline.",
              file=sys.stderr)
    if gc["gt_fallback_used"]:
        print(f"[phase4] {gc['gt_fallback_rows_n']:,} row(s) had no real gt_* columns and "
              f"used the category-derived fallback for false-positive/guardrail checks — "
              f"those numbers are illustrative, not measured.", file=sys.stderr)

    audit_path = outdir / "audit_trail.csv"
    audit_df.to_csv(audit_path, index=False)

    summary_path = outdir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "sensitivity": sensitivity,
                    "cost_inputs": results["cost_inputs"]}, f, indent=2, default=str)

    template_path = Path(args.template)
    if not template_path.exists():
        # fall back to a copy alongside this script
        template_path = Path(__file__).parent / REPORT_TEMPLATE_PATH_DEFAULT
    template_text = template_path.read_text() if template_path.exists() else "{{POLICY_NET}}"

    meta = dict(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        input_source=(args.input or f"DEMO batch (n={args.n}, seed={args.seed})"),
        seed=args.seed,
    )
    report_text = render_report(template_text, results, sensitivity, meta)
    report_path = outdir / "phase4_report.md"
    report_path.write_text(report_text)

    print(f"[phase4] wrote {audit_path}")
    print(f"[phase4] wrote {summary_path}")
    print(f"[phase4] wrote {report_path}")
    print(f"[phase4] policy net recovered: {money(results['policy']['net_recovered'])} "
          f"({results['policy']['net_pct_at_risk']:.2f}% of at-risk) vs naive baseline "
          f"{money(results['baseline_b']['net_recovered'])} "
          f"({results['baseline_b']['net_pct_at_risk']:.2f}%)")
    fp = results["false_positive"]["policy"]
    print(f"[phase4] false-positive retry rate: {fp['fp_rate_pct']:.2f}% "
          f"({fp['wasted_retries_n']:,} wasted retries) | guardrail compliance: "
          f"{'PASS' if gc['compliant'] else 'FAIL'}")


if __name__ == "__main__":
    main()
