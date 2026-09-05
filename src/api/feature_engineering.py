"""
Payment Retry Sequencer — Phase 5: inference-time feature engineering
=======================================================================

`score_with_phase2_model.py` reads the model's feature list off the model
itself (`model.feature_name()`) and expects a batch CSV that has already
been through `build_phase2_features.py`. This API has no such CSV — it
gets one raw request at a time — so this module is the inference-time
equivalent of `build_phase2_features.py`, now that that file has been seen.

Everything below marked CONFIRMED is ported directly from
`build_phase2_features.py`'s Cell-3-equivalent functions (`add_time_features`,
`add_previous_attempt_features`) — same definitions, same edge cases, just
evaluated on one event instead of a DataFrame column. If
`build_phase2_features.py` ever changes, re-port from there, don't hand-edit
here.

Two categories of feature remain NOT fully self-contained:

1. `amount_bucket` — CONFIRMED logic (`pd.cut` with fixed bin edges), but
   the edges themselves are `pd.qcut` quantiles FIT ON THE TRAINING SPLIT
   (`build_phase2_features.py`'s `fit_split_artifacts`), not something any
   single request can supply. These must be loaded once at startup from a
   frozen JSON artifacts file — see `configure_amount_bucket_artifacts()`
   and `main.py`'s lifespan. Run
   `build_phase2_features.py --input <original training csv> --save-artifacts <path>.json`
   once to produce that file.

2. `merchant_archetype` — in training data this was a raw column from
   Phase 1's generator, not derived. This API doesn't receive it directly,
   so it's derived from `merchant_id` by stripping the trailing `_NN`
   suffix (`d2c_ecommerce_01` -> `d2c_ecommerce`), per the merchant-ID
   naming convention. Raises `ValueError` if `merchant_id` doesn't match
   that pattern, rather than guessing.

Everything else falls into:

- DIRECT PASS-THROUGH — fields that exist verbatim on the incoming event
  (`payment_method`, `gateway`, `bank_response_code`, `merchant_id`,
  `amount`, `attempt_number`, `is_recurring`, ...).

- EXPLICIT-OVERRIDE-WITH-FALLBACK — features `main.py`'s
  `PaymentAttemptEvent` accepts as optional fields, used verbatim when the
  caller supplies them (e.g. from a real feature store), with a CONFIRMED
  fallback computed from `timestamp` / `attempt_number` / `previous_attempts`
  when they're not: `is_retry`, `hour`, `is_weekend`, `day_of_month`,
  `is_near_month_end`, `n_prev_attempts_this_txn`, `prev_decline_reason`,
  `prev_bank_response_code`, `day_of_week`, `hour_bucket`.

  `merchant_hist_failure_rate`, `merchant_prior_attempt_count`,
  `customer_hist_failure_rate`, `customer_prior_attempt_count`, and
  `gateway_recent_failure_rate` describe history OUTSIDE this one
  transaction (this merchant/customer/gateway's stats across ALL their
  transactions) — nothing in a single request can substitute for that, so
  these have NO fallback. If omitted, scoring raises `NotImplementedError`
  rather than silently assuming e.g. a 0% historical failure rate, which
  would bias the policy decision. (`build_phase2_features.py` fills
  cold-start cases with a frozen `train_global_fail_rate` rather than
  refusing — if you want that behavior here too instead of a hard
  requirement, that same artifacts file has the number; not wired in yet
  since nothing has asked for it.)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# CONFIRMED against build_phase2_features.py's add_previous_attempt_features —
# sentinel for "there was no previous attempt" in prev_decline_reason /
# prev_bank_response_code. Uppercase "NONE", not "none".
NO_PRIOR_SENTINEL = "NONE"


def _last_prev(previous_attempts: Optional[List[dict]]) -> Optional[dict]:
    if not previous_attempts:
        return None
    return previous_attempts[-1]


def _prev_decline_reason(event: dict, previous_attempts: Optional[List[dict]]) -> str:
    prev = _last_prev(previous_attempts)
    if prev is None:
        return NO_PRIOR_SENTINEL
    if not prev.get("decline_reason_category"):
        raise ValueError(
            "prev_decline_reason: the last entry in previous_attempts has no "
            "decline_reason_category. build_phase2_features.py assumes this "
            "is always present whenever a previous attempt exists — supply "
            "it, matching training-time data."
        )
    return prev["decline_reason_category"]


def _prev_bank_response_code(event: dict, previous_attempts: Optional[List[dict]]) -> str:
    prev = _last_prev(previous_attempts)
    if prev is None:
        return NO_PRIOR_SENTINEL
    if not prev.get("bank_response_code"):
        raise ValueError(
            "prev_bank_response_code: the last entry in previous_attempts has "
            "no bank_response_code. build_phase2_features.py assumes this is "
            "always present whenever a previous attempt exists — supply it, "
            "matching training-time data."
        )
    return prev["bank_response_code"]


def _as_datetime(ts) -> datetime:
    if isinstance(ts, str):
        return datetime.fromisoformat(ts)
    return ts


def _day_of_week(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    # CONFIRMED: build_phase2_features.py uses ts.dt.dayofweek (int, 0=Mon,
    # 6=Sun) — Python's datetime.weekday() uses the identical convention.
    return _as_datetime(event["timestamp"]).weekday()


def _hour_bucket(event: dict, previous_attempts: Optional[List[dict]]) -> str:
    # CONFIRMED against build_phase2_features.py's hour_bucket().
    h = _as_datetime(event["timestamp"]).hour
    if 0 <= h < 5:
        return "night"
    if 5 <= h < 9:
        return "early_morning"
    if 9 <= h < 13:
        return "late_morning"
    if 13 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "late_night"


# ---------------------------------------------------------------------------
# amount_bucket — CONFIRMED pd.cut logic, but bin edges are training-split
# artifacts that must be loaded at startup, not guessed or refit here.
# See build_phase2_features.py's fit_split_artifacts / save_split_artifacts.
# ---------------------------------------------------------------------------

_AMOUNT_BUCKET_ARTIFACTS: Optional[Dict[str, list]] = None


def configure_amount_bucket_artifacts(bin_edges: List[float], bucket_labels: List[str]) -> None:
    """
    Call once at startup (see main.py's lifespan) with the FROZEN,
    training-derived amount_bucket edges/labels — e.g. loaded from the JSON
    file written by `build_phase2_features.py --save-artifacts`. These are
    fit on the training split only and must be reused as-is at inference
    time; refitting from live traffic would silently produce different
    buckets than the model was trained on.
    """
    global _AMOUNT_BUCKET_ARTIFACTS
    if len(bin_edges) != len(bucket_labels) + 1:
        raise ValueError(
            f"amount_bucket artifacts malformed: {len(bin_edges)} bin edges "
            f"but {len(bucket_labels)} label(s) (expected edges == labels + 1)."
        )
    _AMOUNT_BUCKET_ARTIFACTS = {"bin_edges": bin_edges, "bucket_labels": bucket_labels}


def amount_bucket_artifacts_loaded() -> bool:
    return _AMOUNT_BUCKET_ARTIFACTS is not None


def _amount_bucket(event: dict, previous_attempts: Optional[List[dict]]) -> str:
    if _AMOUNT_BUCKET_ARTIFACTS is None:
        raise NotImplementedError(
            "amount_bucket: no frozen bin edges loaded. Run "
            "`build_phase2_features.py --input <original training csv> "
            "--save-artifacts <path>.json` once, then point "
            "AMOUNT_BUCKET_ARTIFACTS_PATH at that file so main.py's startup "
            "can load it."
        )
    amount = event.get("amount")
    if amount is None:
        raise ValueError("amount_bucket: event has no amount to bucket.")

    edges = _AMOUNT_BUCKET_ARTIFACTS["bin_edges"]
    labels = _AMOUNT_BUCKET_ARTIFACTS["bucket_labels"]
    # Matches build_phase2_features.py's featurize():
    #   pd.cut(df["amount"], bins=artifacts["bin_edges"], labels=artifacts["bucket_labels"])
    result = pd.cut([amount], bins=edges, labels=labels)[0]
    if pd.isna(result):
        raise ValueError(
            f"amount_bucket: amount {amount} did not fall into any bucket "
            f"given edges {edges}. The frozen edges' outer bounds should be "
            f"-inf/+inf (build_phase2_features.py guarantees this) — check "
            f"the artifacts file wasn't truncated or mis-saved."
        )
    return str(result)


_MERCHANT_ARCHETYPE_RE = re.compile(r"^(.+)_\d+$")


def _merchant_archetype(event: dict, previous_attempts: Optional[List[dict]]) -> str:
    """
    Phase 1's data generator names merchants '<archetype>_<NN>', e.g.
    'd2c_ecommerce_01', 'saas_subscription_02', 'marketplace_03'. The
    archetype is everything before the trailing '_NN'.
    """
    merchant_id = event.get("merchant_id")
    if not merchant_id:
        raise ValueError("merchant_archetype: event has no merchant_id to derive it from.")

    match = _MERCHANT_ARCHETYPE_RE.match(merchant_id)
    if not match:
        raise ValueError(
            f"merchant_archetype: merchant_id '{merchant_id}' does not match "
            f"the expected '<archetype>_<NN>' pattern (e.g. 'd2c_ecommerce_01'). "
            f"Cannot derive an archetype from it."
        )
    return match.group(1)


def _passthrough(field: str) -> Callable[[dict, Any], Any]:
    def _get(event: dict, previous_attempts: Optional[List[dict]]):
        return event.get(field)
    return _get


def _explicit_or(
    field: str,
    fallback: Optional[Callable[[dict, Optional[List[dict]]], Any]] = None,
    required_msg: Optional[str] = None,
) -> Callable[[dict, Optional[List[dict]]], Any]:
    """
    Prefer the caller-supplied value (`event[field]`, from an optional
    request field of the same name) when present. Otherwise use `fallback`
    if one is safe to compute from request data alone. Otherwise raise —
    on purpose, not a silent guess — with `required_msg` explaining what's
    needed.
    """
    def _get(event: dict, previous_attempts: Optional[List[dict]]):
        val = event.get(field)
        if val is not None:
            return val
        if fallback is not None:
            return fallback(event, previous_attempts)
        raise NotImplementedError(
            required_msg
            or (
                f"'{field}' was not provided in the request and cannot be "
                f"safely derived from request data alone. Supply it "
                f"explicitly (e.g. from your merchant/customer/gateway "
                f"stats store)."
            )
        )
    return _get


# --- fallbacks for the explicit-override-with-fallback features: CONFIRMED
# against build_phase2_features.py's add_time_features / add_previous_
# attempt_features, evaluated on a single event instead of a DataFrame ---

def _fallback_is_retry(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    return int(event.get("attempt_number", 1) > 1)


def _fallback_hour(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    return _as_datetime(event["timestamp"]).hour


def _fallback_is_weekend(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    # CONFIRMED: day_of_week.isin([5, 6]) i.e. Sat/Sun under 0=Mon convention.
    return int(_as_datetime(event["timestamp"]).weekday() in (5, 6))


def _fallback_day_of_month(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    return _as_datetime(event["timestamp"]).day


def _fallback_is_near_month_end(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    # CONFIRMED: (days_in_month - day_of_month) < 4  i.e. the last 4 days
    # of the month (not 3 — verified against build_phase2_features.py).
    import calendar as _calendar
    ts = _as_datetime(event["timestamp"])
    _, days_in_month = _calendar.monthrange(ts.year, ts.month)
    return int((days_in_month - ts.day) < 4)


def _fallback_n_prev_attempts_this_txn(event: dict, previous_attempts: Optional[List[dict]]) -> int:
    return len(previous_attempts) if previous_attempts else 0


# Every feature name the Phase 2 model might ask for that this module knows
# how to build. `main.py` diffs this against `model.feature_name()` at
# startup and refuses to serve traffic if the model needs something not
# listed here (unless ALLOW_UNKNOWN_FEATURES=1 is set).
KNOWN_FEATURE_BUILDERS: Dict[str, Callable[[dict, Optional[List[dict]]], Any]] = {
    "payment_method": _passthrough("payment_method"),
    "gateway": _passthrough("gateway"),
    "bank_response_code": _passthrough("bank_response_code"),
    "merchant_id": _passthrough("merchant_id"),
    "customer_id": _passthrough("customer_id"),
    "amount": _passthrough("amount"),
    "attempt_number": _passthrough("attempt_number"),
    "is_recurring": _passthrough("is_recurring"),
    "prev_decline_reason": _prev_decline_reason,
    "prev_bank_response_code": _prev_bank_response_code,
    "day_of_week": _day_of_week,
    "hour_bucket": _hour_bucket,
    "amount_bucket": _amount_bucket,
    "merchant_archetype": _merchant_archetype,
    "is_retry": _explicit_or("is_retry", _fallback_is_retry),
    "hour": _explicit_or("hour", _fallback_hour),
    "is_weekend": _explicit_or("is_weekend", _fallback_is_weekend),
    "day_of_month": _explicit_or("day_of_month", _fallback_day_of_month),
    "is_near_month_end": _explicit_or("is_near_month_end", _fallback_is_near_month_end),
    "n_prev_attempts_this_txn": _explicit_or(
        "n_prev_attempts_this_txn", _fallback_n_prev_attempts_this_txn
    ),
    "merchant_hist_failure_rate": _explicit_or(
        "merchant_hist_failure_rate",
        required_msg=(
            "'merchant_hist_failure_rate' was not provided. This is this "
            "merchant's historical failure rate across ALL their "
            "transactions, not derivable from this request's "
            "previous_attempts (which only covers this one transaction). "
            "Supply it explicitly, e.g. from a merchant-stats store."
        ),
    ),
    "merchant_prior_attempt_count": _explicit_or(
        "merchant_prior_attempt_count",
        required_msg=(
            "'merchant_prior_attempt_count' was not provided. This is a "
            "count of this merchant's prior attempts across ALL their "
            "transactions, not derivable from this request. Supply it "
            "explicitly, e.g. from a merchant-stats store."
        ),
    ),
    "customer_hist_failure_rate": _explicit_or(
        "customer_hist_failure_rate",
        required_msg=(
            "'customer_hist_failure_rate' was not provided. This is this "
            "customer's historical failure rate across ALL their "
            "transactions, not derivable from this request. Supply it "
            "explicitly, e.g. from a customer-stats store."
        ),
    ),
    "customer_prior_attempt_count": _explicit_or(
        "customer_prior_attempt_count",
        required_msg=(
            "'customer_prior_attempt_count' was not provided. This is a "
            "count of this customer's prior attempts across ALL their "
            "transactions, not derivable from this request. Supply it "
            "explicitly, e.g. from a customer-stats store."
        ),
    ),
    "gateway_recent_failure_rate": _explicit_or(
        "gateway_recent_failure_rate",
        required_msg=(
            "'gateway_recent_failure_rate' was not provided. This is the "
            "gateway's recent failure rate across ALL transactions, not "
            "derivable from this request. Supply it explicitly, e.g. from "
            "a gateway health-monitoring feed."
        ),
    ),
}


def unresolved_features(feature_names: List[str]) -> List[str]:
    """Model-required feature names with NO builder registered at all."""
    return [f for f in feature_names if f not in KNOWN_FEATURE_BUILDERS]


def build_feature_row(
    event: dict, previous_attempts: Optional[List[dict]], feature_names: List[str]
) -> dict:
    """
    Build one row (dict of feature_name -> value) matching the model's
    expected feature list, in whatever order feature_names specifies.
    Raises NotImplementedError (not a silent guess) for any feature this
    module doesn't know how to compute yet, e.g. amount_bucket without
    configured artifacts.
    """
    row = {}
    for f in feature_names:
        builder = KNOWN_FEATURE_BUILDERS.get(f)
        if builder is None:
            raise NotImplementedError(
                f"No feature builder registered for '{f}'. Add one to "
                f"feature_engineering.KNOWN_FEATURE_BUILDERS."
            )
        row[f] = builder(event, previous_attempts)
    return row
