"""
Payment Retry Sequencer — Phase 1: Synthetic Data Generation
==============================================================

Generates a labeled dataset of payment attempt events matching the project's
canonical schema (see schema.json / README). Produces:
    - payment_attempts.csv        (one row per attempt)
    - payment_attempts_sample.json (small human-readable sample)

Design intent (do not break these invariants):
  * `decline_reason_category` is GROUND TRUTH — it is what actually happened,
    not what a model predicted. Phase 2's job is to re-derive it from the
    noisy `bank_response_code`.
  * `risk_block` and `card_expired` are NEVER recoverable, by hard rule.
    This generator enforces that at the data level so that no amount of
    retraining on this data could teach a model otherwise.
  * The `gt_*` (ground truth) columns are for BACKTESTING an ideal policy
    (Phase 4). They must NOT be used as classifier input features — they
    encode information (the true category, the true outcome) that a real
    system would not have at prediction time.
  * `retry_action_taken`, `retry_outcome`, `time_to_recovery` are the
    canonical schema fields owned by the Phase 3 policy engine. Phase 1 data
    is pre-policy history, so these are left null here, exactly as the
    schema specifies ("Null until evaluated").

Run:
    python generate_payment_data.py

Everything tunable lives in CONFIG below.
"""

from __future__ import annotations

import calendar
import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit this to regenerate under different assumptions.
# ---------------------------------------------------------------------------

CONFIG = {
    "seed": 42,
    "start_date": "2025-01-01",
    "end_date": "2025-03-31",          # ~90 days
    "max_attempts_per_intent": 5,       # hard cap on retry-chain length
    "response_code_noise_prob": 0.08,   # P(raw code looks like a DIFFERENT category) — simulates a noisy signal
    "output_dir": "./output",

    # -- Merchant archetypes -------------------------------------------------
    # daily_volume_range = (low, high) successful+failed payment INTENTS
    # per merchant per day, before weekday/hour shaping.
    "merchant_archetypes": {
        "d2c_ecommerce": {
            "n_merchants": 3,
            "daily_volume_range": (40, 120),
            "amount": {"kind": "lognormal", "mean": 1200, "sigma_log": 0.9, "min": 150, "max": 25000},
            "payment_method_mix": {"upi": 0.55, "card": 0.25, "netbanking": 0.08, "wallet": 0.12},
            "is_recurring_rate": 0.02,
            "weekday_multiplier": [1.0, 1.0, 1.0, 1.0, 1.15, 1.4, 1.35],  # Mon..Sun, weekend spike
            "hour_weights": [1,1,1,1,1,2,4,6,7,8,8,9,9,9,8,8,9,10,11,12,10,7,4,2],  # evening peak
            "n_customers_multiplier": 25,  # ~volume * this = distinct customer pool size
        },
        "saas_subscription": {
            "n_merchants": 2,
            "daily_volume_range": (20, 60),
            "amount": {"kind": "choice", "values": [499, 999, 1999, 2999, 4999, 9999],
                       "weights": [0.30, 0.28, 0.20, 0.12, 0.07, 0.03]},
            "payment_method_mix": {"card": 0.65, "upi": 0.30, "netbanking": 0.03, "wallet": 0.02},
            "is_recurring_rate": 0.92,
            "weekday_multiplier": [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5],  # weekday-heavy (billing runs)
            "hour_weights": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # overridden for recurring below
            # Recurring/subscription billing in India is dominated by scheduled
            # bank-side batch runs, which commonly execute overnight. This is
            # a modeling assumption (see assumptions.md), not a cited stat.
            "recurring_billing_hour_weights": [10,10,9,7,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,4,7],
            "n_customers_multiplier": 8,   # subscriptions = smaller, repeating customer pool
        },
        "marketplace": {
            "n_merchants": 3,
            "daily_volume_range": (50, 150),
            "amount": {"kind": "lognormal", "mean": 900, "sigma_log": 1.1, "min": 100, "max": 50000},
            "payment_method_mix": {"upi": 0.45, "card": 0.20, "netbanking": 0.15, "wallet": 0.20},
            "is_recurring_rate": 0.01,
            "weekday_multiplier": [1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2],
            "hour_weights": [1,1,1,1,1,2,3,5,7,8,9,10,10,9,8,8,9,10,11,11,9,6,3,2],
            "n_customers_multiplier": 20,
        },
    },

    "gateways": ["gateway_a", "gateway_b", "gateway_c"],

    # -- Decline calibration --------------------------------------------------
    # Overall failure rates and the mix of reasons behind them are loosely
    # calibrated against public reporting (see assumptions.md / README for
    # exact sourcing and, importantly, where I could NOT find a public number
    # and had to assume one). These are deliberately editable.
    "decline": {
        "base_failure_rate": {
            "upi": 0.06,
            "card": 0.10,
            "netbanking": 0.09,
            "wallet": 0.04,
        },
        "reason_distribution": {
            "upi": {
                "insufficient_funds": 0.35, "invalid_otp": 0.15, "limit_exceeded": 0.08,
                "risk_block": 0.05, "bank_server_error": 0.20, "network_timeout": 0.12,
                "other": 0.05,
            },
            "card": {
                "insufficient_funds": 0.30, "card_expired": 0.12, "risk_block": 0.10,
                "invalid_otp": 0.15, "limit_exceeded": 0.08, "bank_server_error": 0.15,
                "network_timeout": 0.05, "other": 0.05,
            },
            "netbanking": {
                "insufficient_funds": 0.20, "bank_server_error": 0.30, "network_timeout": 0.25,
                "invalid_otp": 0.10, "risk_block": 0.05, "limit_exceeded": 0.05, "other": 0.05,
            },
            "wallet": {
                "insufficient_funds": 0.45, "limit_exceeded": 0.20, "invalid_otp": 0.05,
                "bank_server_error": 0.10, "network_timeout": 0.05, "risk_block": 0.05,
                "other": 0.10,
            },
        },
    },

    # -- Time-of-day / calendar effects ---------------------------------------
    "time_effects": {
        "late_night_hours": (22, 24),      # 22:00–23:59: batch-job / bill-pay spike window
        "late_night_multiplier": 1.8,
        "month_end_days": 4,                # last N days of month: bank load + pre-salary crunch
        "month_end_multiplier": 1.6,
        "maintenance_windows": {            # {gateway: (start_hour, end_hour)}
            "gateway_b": (2, 4),
            "gateway_c": (3, 4),
        },
        "maintenance_multiplier": 5.0,
        "monday_morning_multiplier": 1.3,   # pent-up demand, first business day effect (simplified to every Monday)
        "gateway_degraded_day_prob": 0.03,  # per gateway, per day: independent "bad day" (outage/degradation)
        "gateway_degraded_multiplier": 2.0,
    },

    # Common salary-credit days used to model insufficient_funds recoverability.
    "salary_days": [1, 2, 3, 7],

    # -- Naive/historical retry behavior (NOT the ideal policy) ---------------
    # This models what an *unmanaged* system does today: some blind retrying
    # regardless of whether it's a good idea, at essentially arbitrary delays.
    # This is intentionally worse than the `gt_*` ideal policy — Phase 4's
    # backtest is the gap between this baseline and the ideal.
    "naive_retry": {
        "probability_by_reason": {
            "insufficient_funds": 0.50, "bank_server_error": 0.60, "invalid_otp": 0.55,
            "network_timeout": 0.60, "limit_exceeded": 0.45, "risk_block": 0.30,
            "card_expired": 0.40, "other": 0.35,
        },
        "delay_seconds_range": {  # naive systems don't know the "right" wait time
            "insufficient_funds": (60, 6 * 3600),
            "bank_server_error": (30, 1800),
            "invalid_otp": (20, 900),
            "network_timeout": (5, 300),
            "limit_exceeded": (60, 3600),
            "risk_block": (60, 3600),
            "card_expired": (3600, 3 * 86400),
            "other": (60, 7200),
        },
    },

    # -- Raw bank_response_code pools per payment_method -----------------------
    # IMPORTANT: these are SYNTHETIC / illustrative codes, not a verified real
    # NPCI or bank-issued code list (issuers and gateways don't publish a
    # single canonical list; codes vary bank-to-bank in reality — which is
    # itself part of why the raw signal is "noisy" in the real world). The
    # `card` pool loosely mirrors commonly-documented ISO 8583-style decline
    # codes (see assumptions.md for the source). UPI/netbanking/wallet codes
    # are invented placeholders with a consistent, documented naming scheme.
    "response_code_pools": {
        "card": {
            "insufficient_funds": ["51"],
            "card_expired": ["54"],
            "risk_block": ["07", "41", "59", "05"],
            "invalid_otp": ["3DS_OTP_FAIL", "80"],
            "limit_exceeded": ["61", "65"],
            "bank_server_error": ["91", "96"],
            "network_timeout": ["68", "91"],
            "other": ["30", "GENERIC_DECLINE"],
        },
        "upi": {
            "insufficient_funds": ["U69"],
            "invalid_otp": ["U58", "U96"],
            "risk_block": ["U39"],
            "limit_exceeded": ["U51"],
            "bank_server_error": ["UBT_DOWN", "U96"],
            "network_timeout": ["UBT_TIMEOUT", "U01"],
            "card_expired": [],
            "other": ["UZZ"],
        },
        "netbanking": {
            "insufficient_funds": ["NB_INSUFF"],
            "invalid_otp": ["NB_OTP_FAIL"],
            "risk_block": ["NB_RISK"],
            "limit_exceeded": ["NB_LIMIT"],
            "bank_server_error": ["NB_BANK_DOWN"],
            "network_timeout": ["NB_TIMEOUT", "NB_REDIRECT_FAIL"],
            "card_expired": [],
            "other": ["NB_UNKNOWN"],
        },
        "wallet": {
            "insufficient_funds": ["WL_LOW_BAL"],
            "limit_exceeded": ["WL_KYC_LIMIT"],
            "invalid_otp": ["WL_OTP_FAIL"],
            "bank_server_error": ["WL_SERVER_ERR"],
            "network_timeout": ["WL_TIMEOUT"],
            "risk_block": ["WL_RISK"],
            "card_expired": [],
            "other": ["WL_UNKNOWN"],
        },
    },
    "success_code": "00",
}

DECLINE_REASONS = [
    "insufficient_funds", "bank_server_error", "invalid_otp", "risk_block",
    "network_timeout", "card_expired", "limit_exceeded", "other",
]
RETRY_ACTIONS = [
    "same_gateway_retry", "alt_gateway_retry", "alt_payment_method_prompt",
    "delayed_retry", "no_retry",
]
NON_RECOVERABLE = {"risk_block", "card_expired", "other"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def weighted_choice(dist: dict, rng: np.random.Generator):
    keys = list(dist.keys())
    weights = np.array(list(dist.values()), dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(keys), p=weights)
    return keys[idx]


def sample_hour(weights: list[float], rng: np.random.Generator) -> int:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return int(rng.choice(24, p=w))


def days_until_next_salary_day(ts: datetime, salary_days: list[int]) -> int:
    """Days from ts until the next date whose day-of-month is in salary_days."""
    d = ts.date()
    candidates = sorted(salary_days)
    for day in candidates:
        last_dom = calendar.monthrange(d.year, d.month)[1]
        if day > last_dom:
            continue
        cand = date(d.year, d.month, day)
        if cand >= d:
            return (cand - d).days
    # roll to next month
    ny, nm = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    for day in candidates:
        last_dom = calendar.monthrange(ny, nm)[1]
        if day > last_dom:
            continue
        cand = date(ny, nm, day)
        return (cand - d).days
    return 30  # fallback, should not happen with default config


def sample_amount(spec: dict, rng: np.random.Generator) -> int:
    if spec["kind"] == "lognormal":
        mu = np.log(spec["mean"])
        val = rng.lognormal(mean=mu, sigma=spec["sigma_log"])
        val = float(np.clip(val, spec["min"], spec["max"]))
        return int(round(val / 5) * 5)  # round to nearest 5 rupees, realistic ticket sizes
    elif spec["kind"] == "choice":
        return int(rng.choice(spec["values"], p=spec["weights"]))
    raise ValueError(f"Unknown amount spec kind: {spec['kind']}")


def sample_payment_method(mix: dict, rng: np.random.Generator) -> str:
    return weighted_choice(mix, rng)


# ---------------------------------------------------------------------------
# Context / decline-probability model
# ---------------------------------------------------------------------------


def compute_context(ts: datetime, gateway: str, gateway_degraded: bool, config: dict) -> dict:
    te = config["time_effects"]
    hour = ts.hour
    last_dom = calendar.monthrange(ts.year, ts.month)[1]
    is_month_end = (last_dom - ts.day) < te["month_end_days"]
    is_late_night = te["late_night_hours"][0] <= hour < te["late_night_hours"][1]
    maint = te["maintenance_windows"].get(gateway)
    is_maintenance = bool(maint and maint[0] <= hour < maint[1])
    is_monday_morning = ts.weekday() == 0 and 9 <= hour < 11
    return {
        "is_month_end": is_month_end,
        "is_late_night": is_late_night,
        "is_maintenance": is_maintenance,
        "is_monday_morning": is_monday_morning,
        "gateway_degraded": gateway_degraded,
    }


def failure_multiplier(context: dict, config: dict) -> float:
    te = config["time_effects"]
    m = 1.0
    if context["is_late_night"]:
        m *= te["late_night_multiplier"]
    if context["is_month_end"]:
        m *= te["month_end_multiplier"]
    if context["is_maintenance"]:
        m *= te["maintenance_multiplier"]
    if context["is_monday_morning"]:
        m *= te["monday_morning_multiplier"]
    if context["gateway_degraded"]:
        m *= te["gateway_degraded_multiplier"]
    return m


def adjust_reason_distribution(base_dist: dict, context: dict) -> dict:
    dist = dict(base_dist)
    infra_boost = 1.0
    if context["is_maintenance"]:
        infra_boost *= 4.0
    if context["gateway_degraded"]:
        infra_boost *= 2.0
    if context["is_late_night"]:
        infra_boost *= 1.5
    if infra_boost > 1.0:
        for k in ("bank_server_error", "network_timeout"):
            if k in dist:
                dist[k] *= infra_boost
    if context["is_month_end"]:
        if "insufficient_funds" in dist:
            dist["insufficient_funds"] *= 1.3
        if "bank_server_error" in dist:
            dist["bank_server_error"] *= 1.2
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def sample_bank_response_code(category: str, method: str, rng: np.random.Generator, config: dict) -> str:
    pools = config["response_code_pools"][method]
    noise_prob = config["response_code_noise_prob"]
    chosen_cat = category
    if rng.random() < noise_prob:
        alt_cats = [c for c, codes in pools.items() if c != category and codes]
        if alt_cats:
            chosen_cat = alt_cats[rng.integers(len(alt_cats))]
    codes = pools.get(chosen_cat) or pools.get(category) or ["UNK"]
    return codes[rng.integers(len(codes))]


# ---------------------------------------------------------------------------
# Ground-truth "ideal retry policy" oracle (for backtesting, Phase 4 only)
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    is_recoverable: bool
    ideal_action: str | None
    ideal_time_to_recovery_seconds: float | None
    ideal_timing_hint: str | None
    requires_customer_action: bool = False  # flags invalid_otp — see assumptions.md


def compute_ground_truth(category: str, ts: datetime, gateway_degraded: bool,
                          config: dict, rng: np.random.Generator) -> GroundTruth:
    if category in ("risk_block", "card_expired", "other"):
        return GroundTruth(False, "no_retry", None, None)

    if category == "network_timeout":
        recoverable = rng.random() < 0.95
        ttr = float(rng.uniform(5, 120)) if recoverable else None
        return GroundTruth(recoverable, "same_gateway_retry", ttr, None)

    if category == "bank_server_error":
        if gateway_degraded:
            recoverable = rng.random() < 0.60
            ttr = float(rng.uniform(300, 3600)) if recoverable else None
            return GroundTruth(recoverable, "alt_gateway_retry", ttr, None)
        recoverable = rng.random() < 0.85
        ttr = float(rng.uniform(30, 1800)) if recoverable else None
        return GroundTruth(recoverable, "same_gateway_retry", ttr, None)

    if category == "invalid_otp":
        recoverable = rng.random() < 0.70
        ttr = float(rng.uniform(30, 600)) if recoverable else None
        # NOTE: the project's retry-action taxonomy doesn't define a distinct
        # action for "customer-initiated fresh attempt" — see assumptions.md.
        # We reuse same_gateway_retry and flag it, rather than silently
        # inventing a new taxonomy value.
        return GroundTruth(recoverable, "same_gateway_retry", ttr, None, requires_customer_action=True)

    if category == "limit_exceeded":
        recoverable = rng.random() < 0.80
        ttr = float(rng.uniform(60, 1800)) if recoverable else None
        return GroundTruth(recoverable, "alt_payment_method_prompt", ttr, None)

    if category == "insufficient_funds":
        days_to_salary = days_until_next_salary_day(ts, config["salary_days"])
        base_prob = 0.75 if days_to_salary <= 3 else 0.35
        recoverable = rng.random() < base_prob
        ttr = float(days_to_salary * 86400 + rng.uniform(0, 7200)) if recoverable else None
        hint = f"retry in ~{days_to_salary} day(s) (next likely salary-credit window)"
        return GroundTruth(recoverable, "delayed_retry", ttr, hint)

    return GroundTruth(False, "no_retry", None, None)  # should be unreachable


# ---------------------------------------------------------------------------
# Merchant setup
# ---------------------------------------------------------------------------


@dataclass
class Merchant:
    merchant_id: str
    archetype: str
    spec: dict
    customer_pool_size: int


def build_merchants(config: dict) -> list[Merchant]:
    merchants = []
    for archetype, spec in config["merchant_archetypes"].items():
        avg_daily = sum(spec["daily_volume_range"]) / 2
        pool_size = max(50, int(avg_daily * spec["n_customers_multiplier"]))
        for i in range(spec["n_merchants"]):
            merchants.append(Merchant(
                merchant_id=f"{archetype}_{i+1:02d}",
                archetype=archetype,
                spec=spec,
                customer_pool_size=pool_size,
            ))
    return merchants


# ---------------------------------------------------------------------------
# Core simulation: one payment "intent" -> chain of 1..N attempts
# ---------------------------------------------------------------------------


def simulate_intent(merchant: Merchant, day: date, config: dict, rng: np.random.Generator,
                     gateway_degraded_lookup: dict) -> list[dict]:
    spec = merchant.spec
    method = sample_payment_method(spec["payment_method_mix"], rng)
    gateway = config["gateways"][rng.integers(len(config["gateways"]))]
    amount = sample_amount(spec["amount"], rng)
    is_recurring = rng.random() < spec["is_recurring_rate"]
    customer_id = f"cust_{merchant.merchant_id}_{rng.integers(merchant.customer_pool_size):06d}"

    if is_recurring and "recurring_billing_hour_weights" in spec:
        hour = sample_hour(spec["recurring_billing_hour_weights"], rng)
    else:
        hour = sample_hour(spec["hour_weights"], rng)
    minute, second = int(rng.integers(60)), int(rng.integers(60))
    ts = datetime(day.year, day.month, day.day, hour, minute, second)

    session_id = f"{merchant.merchant_id}-{day.isoformat()}-{uuid.uuid4().hex[:8]}"
    attempts = []
    prev_meta = []
    sticky_category = None  # for card_expired / risk_block: doesn't resolve itself

    for attempt_number in range(1, config["max_attempts_per_intent"] + 1):
        degraded = gateway_degraded_lookup.get((ts.date(), gateway), False)
        context = compute_context(ts, gateway, degraded, config)

        if sticky_category is not None:
            failed = True
            category = sticky_category
        else:
            p_fail = min(0.95, config["decline"]["base_failure_rate"][method] * failure_multiplier(context, config))
            failed = rng.random() < p_fail
            category = None
            if failed:
                dist = adjust_reason_distribution(config["decline"]["reason_distribution"][method], context)
                category = weighted_choice(dist, rng)

        transaction_id = uuid.uuid4().hex

        if not failed:
            row = {
                "transaction_id": transaction_id,
                "payment_session_id": session_id,  # extension beyond canonical schema — see assumptions.md
                "merchant_id": merchant.merchant_id,
                "merchant_archetype": merchant.archetype,  # extension, useful for stratified analysis
                "customer_id": customer_id,
                "amount": amount,
                "currency": "INR",
                "payment_method": method,
                "gateway": gateway,
                "bank_response_code": config["success_code"],
                "decline_reason_category": None,
                "timestamp": ts.isoformat(),
                "attempt_number": attempt_number,
                "is_recurring": bool(is_recurring),
                "previous_attempts": json.dumps(prev_meta),
                "retry_action_taken": None,
                "retry_outcome": None,
                "time_to_recovery": None,
                "success": True,
                "gt_is_recoverable": None,
                "gt_ideal_retry_action": None,
                "gt_ideal_time_to_recovery_seconds": None,
                "gt_ideal_timing_hint": None,
                "gt_requires_customer_action": None,
                "ctx_is_month_end": context["is_month_end"],
                "ctx_is_late_night": context["is_late_night"],
                "ctx_is_maintenance_window": context["is_maintenance"],
                "ctx_gateway_degraded_day": context["gateway_degraded"],
            }
            attempts.append(row)
            break

        code = sample_bank_response_code(category, method, rng, config)
        gt = compute_ground_truth(category, ts, context["gateway_degraded"], config, rng)

        row = {
            "transaction_id": transaction_id,
            "payment_session_id": session_id,
            "merchant_id": merchant.merchant_id,
            "merchant_archetype": merchant.archetype,
            "customer_id": customer_id,
            "amount": amount,
            "currency": "INR",
            "payment_method": method,
            "gateway": gateway,
            "bank_response_code": code,
            "decline_reason_category": category,
            "timestamp": ts.isoformat(),
            "attempt_number": attempt_number,
            "is_recurring": bool(is_recurring),
            "previous_attempts": json.dumps(prev_meta),
            "retry_action_taken": None,
            "retry_outcome": None,
            "time_to_recovery": None,
            "success": False,
            "gt_is_recoverable": gt.is_recoverable,
            "gt_ideal_retry_action": gt.ideal_action,
            "gt_ideal_time_to_recovery_seconds": gt.ideal_time_to_recovery_seconds,
            "gt_ideal_timing_hint": gt.ideal_timing_hint,
            "gt_requires_customer_action": gt.requires_customer_action,
            "ctx_is_month_end": context["is_month_end"],
            "ctx_is_late_night": context["is_late_night"],
            "ctx_is_maintenance_window": context["is_maintenance"],
            "ctx_gateway_degraded_day": context["gateway_degraded"],
        }
        attempts.append(row)
        prev_meta.append({
            "transaction_id": transaction_id,
            "attempt_number": attempt_number,
            "timestamp": ts.isoformat(),
            "bank_response_code": code,
            "decline_reason_category": category,
            "success": False,
        })

        if category in ("card_expired", "risk_block"):
            sticky_category = category  # a naive system may still retry, but it will keep failing the same way

        naive_prob = config["naive_retry"]["probability_by_reason"].get(category, 0.3)
        will_retry = rng.random() < naive_prob and attempt_number < config["max_attempts_per_intent"]
        if not will_retry:
            break
        lo, hi = config["naive_retry"]["delay_seconds_range"].get(category, (60, 3600))
        delay = float(rng.uniform(lo, hi))
        ts = ts + timedelta(seconds=delay)

    return attempts


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_dataset(config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(config["seed"])
    start = date.fromisoformat(config["start_date"])
    end = date.fromisoformat(config["end_date"])
    merchants = build_merchants(config)

    # Pre-roll gateway "bad day" states so degradation is a per-day, per-gateway
    # fact (not re-rolled per transaction, which wouldn't be realistic).
    gateway_degraded_lookup = {}
    for day in daterange(start, end):
        for gw in config["gateways"]:
            gateway_degraded_lookup[(day, gw)] = rng.random() < config["time_effects"]["gateway_degraded_day_prob"]

    all_rows = []
    for day in daterange(start, end):
        weekday = day.weekday()
        for merchant in merchants:
            spec = merchant.spec
            lo, hi = spec["daily_volume_range"]
            base_n = rng.integers(lo, hi + 1)
            n_intents = int(round(base_n * spec["weekday_multiplier"][weekday]))
            for _ in range(n_intents):
                all_rows.extend(simulate_intent(merchant, day, config, rng, gateway_degraded_lookup))

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    n_failed = int((~df["success"]).sum())
    print(f"Total attempt rows:        {n:,}")
    print(f"Failed attempts:           {n_failed:,} ({n_failed/n:.1%})")
    print(f"Distinct payment sessions: {df['payment_session_id'].nunique():,}")
    print()
    print("Failure rate by payment_method:")
    print((1 - df.groupby("payment_method")["success"].mean()).round(3).to_string())
    print()
    print("decline_reason_category distribution (failed attempts only):")
    print(df.loc[~df["success"], "decline_reason_category"].value_counts(normalize=True).round(3).to_string())
    print()
    failed = df.loc[~df["success"]]
    print("Ground-truth recoverability by category:")
    print(failed.groupby("decline_reason_category")["gt_is_recoverable"].mean().round(3).to_string())
    non_recov = failed[failed["decline_reason_category"].isin(NON_RECOVERABLE)]
    bad = non_recov["gt_is_recoverable"].sum()
    print()
    print(f"Sanity check — risk_block/card_expired/other marked recoverable: {int(bad)} (must be 0)")


def main():
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(CONFIG)
    print_summary(df)

    csv_path = out_dir / "payment_attempts.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df):,} rows)")

    sample_path = out_dir / "payment_attempts_sample.json"
    sample = json.loads(df.head(20).to_json(orient="records", date_format="iso"))
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2, default=str)
    print(f"Wrote {sample_path}")

    config_path = out_dir / "generation_config_used.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
