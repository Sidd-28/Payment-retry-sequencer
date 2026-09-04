"""
build_phase2_features.py — reproduce train_classifier.py's Cells 2-4 on a raw
payment_attempts.csv, so score_with_phase2_model.py stops failing on the 16
engineered columns that only ever existed inside train_classifier.py's own
process memory (amount_bucket, is_retry, hour, hour_bucket, day_of_week,
is_weekend, day_of_month, is_near_month_end, n_prev_attempts_this_txn,
prev_decline_reason, prev_bank_response_code, merchant_hist_failure_rate,
merchant_prior_attempt_count, customer_hist_failure_rate,
customer_prior_attempt_count, gateway_recent_failure_rate).

    python build_phase2_features.py \\
        --input data/payment_attempts.csv \\
        --output data/payment_attempts_featured.csv

Then:
    python score_with_phase2_model.py --input data/payment_attempts_featured.csv

--------------------------------------------------------------------------
NOT A RE-DERIVATION
--------------------------------------------------------------------------
add_time_features / add_previous_attempt_features / add_rolling_history_features
below are copied unchanged from train_classifier.py Cell 3 — same code, same
column names, same order. FEATURE_COLS is copied unchanged from Cell 5. If
train_classifier.py ever changes, re-copy from there, don't hand-edit here.

--------------------------------------------------------------------------
ONE THING THIS CANNOT FAKE — READ BEFORE TRUSTING THE OUTPUT
--------------------------------------------------------------------------
Two pieces of Cell 4 are FIT on the training split, not computed per-row, and
train_classifier.py never saved them to disk anywhere:

  * amount_bucket edges       — pd.qcut(train_amounts, q=5), train_amounts
                                 being rows with timestamp < the 70th-
                                 percentile timestamp of the WHOLE file
  * train_global_fail_rate    — mean(failed) over that same train slice,
                                 used to fill cold-start merchant/customer/
                                 gateway history features

By default this script recomputes both from whatever --input you give it, the
same way Cell 4 does. That reproduces train_classifier.py's output EXACTLY
if, and only if, --input is the same file (same rows, same timestamp range)
training ran on — recomputing a quantile over the same data gives the same
number every time.

If you ever run this against a genuinely NEW batch (e.g. a future day's
attempts, not present in the file training used), recomputing these two
values from that new batch's own distribution will NOT match training — you'd
be fitting bucket edges to a different set of amounts than the model learned
against. For that case: run once with --save-artifacts against the ORIGINAL
training CSV, then pass --artifacts-in on every later run against new
batches, so the frozen, training-derived numbers get reused instead of
refit each time.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Verbatim from train_classifier.py Cell 3.
# ---------------------------------------------------------------------------

def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    ts = frame["timestamp"]
    frame["hour"] = ts.dt.hour
    frame["day_of_week"] = ts.dt.dayofweek  # 0=Mon
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["day_of_month"] = ts.dt.day
    days_in_month = ts.dt.days_in_month
    frame["is_near_month_end"] = ((days_in_month - frame["day_of_month"]) < 4).astype(int)

    def hour_bucket(h):
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

    frame["hour_bucket"] = frame["hour"].apply(hour_bucket)
    return frame


def add_previous_attempt_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    def parse(prev_json):
        try:
            items = json.loads(prev_json) if isinstance(prev_json, str) else []
        except (json.JSONDecodeError, TypeError):
            items = []
        return items

    parsed = frame["previous_attempts"].apply(parse)
    frame["n_prev_attempts_this_txn"] = parsed.apply(len)
    frame["prev_decline_reason"] = parsed.apply(
        lambda items: items[-1]["decline_reason_category"] if items else "NONE"
    )
    frame["prev_bank_response_code"] = parsed.apply(
        lambda items: items[-1]["bank_response_code"] if items else "NONE"
    )
    frame["is_retry"] = (frame["attempt_number"] > 1).astype(int)
    return frame


def add_rolling_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Expanding/rolling stats computed strictly on PRIOR rows (shift(1)),
    over the whole time-sorted frame. Must be called on the full dataset
    (successes included) BEFORE any filtering — that's what makes it
    leakage-safe: each row only ever sees the past."""
    frame = frame.copy()
    frame["failed"] = (~frame["success"]).astype(int)

    for key, prefix in [("merchant_id", "merchant"), ("customer_id", "customer")]:
        grp = frame.groupby(key)["failed"]
        prior_count = grp.cumcount()
        prior_fail_sum = grp.cumsum() - frame["failed"]
        frame[f"{prefix}_prior_attempt_count"] = prior_count
        frame[f"{prefix}_hist_failure_rate"] = np.where(
            prior_count > 0, prior_fail_sum / prior_count.replace(0, np.nan), np.nan
        )

    frame["gateway_recent_failure_rate"] = (
        frame.groupby("gateway")["failed"]
        .transform(lambda s: s.rolling(window=300, min_periods=10).mean().shift(1))
    )
    return frame


# Copied from train_classifier.py Cell 4.
FILLNA_COLS = ["merchant_hist_failure_rate", "customer_hist_failure_rate", "gateway_recent_failure_rate"]

# Copied from train_classifier.py Cell 5 — the 24 columns score_with_phase2_model.py needs.
FEATURE_COLS = [
    "payment_method", "gateway", "bank_response_code", "merchant_archetype", "merchant_id",
    "amount", "amount_bucket",
    "attempt_number", "is_retry",
    "is_recurring",
    "hour", "hour_bucket", "day_of_week", "is_weekend", "day_of_month", "is_near_month_end",
    "n_prev_attempts_this_txn", "prev_decline_reason", "prev_bank_response_code",
    "merchant_hist_failure_rate", "merchant_prior_attempt_count",
    "customer_hist_failure_rate", "customer_prior_attempt_count",
    "gateway_recent_failure_rate",
]

REQUIRED_RAW_COLUMNS = {
    "timestamp", "success", "merchant_id", "customer_id", "gateway", "amount",
    "previous_attempts", "attempt_number", "payment_method", "bank_response_code",
    "merchant_archetype", "is_recurring",
}


# ---------------------------------------------------------------------------
# Split-fit artifacts (amount_bucket edges + train_global_fail_rate)
# ---------------------------------------------------------------------------

def fit_split_artifacts(df: pd.DataFrame) -> dict:
    """Reproduces train_classifier.py Cell 4's train-only fitting, using
    `timestamp < 70th-percentile timestamp` as the definition of the train
    slice (equivalent to Cell 4's `split == "train"`, since val_cut never
    affects which rows count as "train")."""
    train_cut = df["timestamp"].quantile(0.70)
    train_mask = df["timestamp"] < train_cut

    train_amounts = df.loc[train_mask, "amount"]
    _, bin_edges = pd.qcut(train_amounts, q=5, retbins=True, duplicates="drop")
    bin_edges = np.asarray(bin_edges, dtype=float).copy()
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    bucket_labels = [f"q{i + 1}" for i in range(len(bin_edges) - 1)]

    train_global_fail_rate = float(df.loc[train_mask, "failed"].mean())

    print(f"[featurize] fit amount_bucket + train_global_fail_rate from this file's own "
          f"70th-percentile timestamp cutoff ({train_cut}, {train_mask.sum():,} train rows). "
          f"This matches train_classifier.py's Cell 4 EXACTLY only if --input is the same file "
          f"training ran on.", file=sys.stderr)

    return {
        "bin_edges": bin_edges,
        "bucket_labels": bucket_labels,
        "train_global_fail_rate": train_global_fail_rate,
        "train_cut": str(train_cut),
    }


def save_split_artifacts(path: Path, artifacts: dict) -> None:
    payload = {
        "amount_bucket_inner_edges": [float(x) for x in artifacts["bin_edges"][1:-1]],
        "amount_bucket_labels": artifacts["bucket_labels"],
        "train_global_fail_rate": artifacts["train_global_fail_rate"],
        "fit_from_train_cut_timestamp": artifacts["train_cut"],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[featurize] saved split artifacts -> {path} (reuse via --artifacts-in on future batches)",
          file=sys.stderr)


def load_split_artifacts(path: Path) -> dict:
    payload = json.loads(path.read_text())
    inner = payload["amount_bucket_inner_edges"]
    bin_edges = np.array([-np.inf, *inner, np.inf])
    print(f"[featurize] loaded frozen split artifacts from {path} "
          f"(originally fit from train_cut={payload.get('fit_from_train_cut_timestamp')}) "
          f"instead of refitting from --input.", file=sys.stderr)
    return {
        "bin_edges": bin_edges,
        "bucket_labels": payload["amount_bucket_labels"],
        "train_global_fail_rate": payload["train_global_fail_rate"],
    }


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def featurize(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)  # Cell 2

    df = add_time_features(df)             # Cell 3
    df = add_previous_attempt_features(df)  # Cell 3
    df = add_rolling_history_features(df)   # Cell 3 — needs "failed", added above

    df["amount_bucket"] = pd.cut(df["amount"], bins=artifacts["bin_edges"], labels=artifacts["bucket_labels"])
    return df


def apply_cold_start_fillna(frame: pd.DataFrame, train_global_fail_rate: float) -> pd.DataFrame:
    frame = frame.copy()
    for col in FILLNA_COLS:
        frame[col] = frame[col].fillna(train_global_fail_rate)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Reproduce train_classifier.py's feature engineering on a raw CSV.")
    ap.add_argument("--input", type=str, required=True, help="Raw payment_attempts.csv.")
    ap.add_argument("--output", type=str, default=None,
                     help="Where to write the featured CSV. Defaults to <input>_featured.csv (never overwrites --input).")
    ap.add_argument("--save-artifacts", type=str, default=None,
                     help="Write the fitted amount_bucket edges + train_global_fail_rate to this JSON path, "
                          "for reuse via --artifacts-in on future batches. Recommended on your first run, "
                          "against the original training CSV.")
    ap.add_argument("--artifacts-in", type=str, default=None,
                     help="Load amount_bucket edges + train_global_fail_rate from a JSON file written by a "
                          "prior --save-artifacts run, instead of refitting them from --input. Use this for "
                          "any batch that is NOT the original training CSV.")
    ap.add_argument("--keep-all-rows", action="store_true",
                     help="Keep successful (non-declined) rows too. Off by default: train_classifier.py only "
                          "ever engineered/labeled the (success == False) subset — there's no decline_reason "
                          "to predict for a successful attempt.")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"--input {input_path} does not exist.")
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_featured.csv")
    if output_path.resolve() == input_path.resolve():
        raise SystemExit("--output must not be the same file as --input.")

    artifacts_in_path = Path(args.artifacts_in) if args.artifacts_in else None
    if artifacts_in_path and not artifacts_in_path.exists():
        raise SystemExit(f"--artifacts-in {artifacts_in_path} does not exist.")

    df = pd.read_csv(input_path)
    print(f"[featurize] loaded {len(df):,} rows from {input_path}", file=sys.stderr)

    missing_raw = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing_raw:
        raise SystemExit(f"--input is missing raw column(s) this script needs: {sorted(missing_raw)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_for_fit = df.sort_values("timestamp").reset_index(drop=True)
    df_for_fit["failed"] = (~df_for_fit["success"]).astype(int)  # only needed if fitting fresh

    artifacts = (
        load_split_artifacts(artifacts_in_path)
        if artifacts_in_path
        else fit_split_artifacts(df_for_fit)
    )
    if args.save_artifacts:
        if artifacts_in_path:
            raise SystemExit("--save-artifacts and --artifacts-in together would just re-save what you "
                              "loaded — pick one: fit + save (first run) or load (later runs).")
        save_split_artifacts(Path(args.save_artifacts), artifacts)

    featured = featurize(df, artifacts)

    if args.keep_all_rows:
        out = featured
        print(f"[featurize] --keep-all-rows set: keeping all {len(out):,} rows, including successful "
              f"attempts. predicted_category on those rows is not meaningful.", file=sys.stderr)
    else:
        out = featured[~featured["success"]].copy()
        print(f"[featurize] kept {len(out):,} of {len(featured):,} rows (success == False — matches "
              f"train_classifier.py's `labeled` subset). Pass --keep-all-rows to keep everything.",
              file=sys.stderr)

    out = apply_cold_start_fillna(out, artifacts["train_global_fail_rate"])

    missing_features = [c for c in FEATURE_COLS if c not in out.columns]
    if missing_features:
        raise SystemExit(f"Internal error — still missing engineered column(s): {missing_features}")
    print(f"[featurize] all {len(FEATURE_COLS)} FEATURE_COLS present.", file=sys.stderr)

    out.to_csv(output_path, index=False)
    print(f"\n[featurize] wrote {output_path}", file=sys.stderr)
    print(f"[featurize] next: python score_with_phase2_model.py --input {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
