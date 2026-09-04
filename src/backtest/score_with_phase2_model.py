"""
Phase 2 model scoring — produce predicted_category / predicted_confidence
===========================================================================

Loads the trained Phase 2 decline-reason classifier and runs it over a raw
payment-attempts CSV, appending the two columns phase4_backtest.py looks
for (`predicted_category`, `predicted_confidence`) so the backtest uses
your REAL model output instead of simulating it from the confusion matrix.

If your raw CSV doesn't already have the 24 columns this model expects
(check with the error this script raises if any are missing), run
build_phase2_features.py first:

    python build_phase2_features.py \\
        --input data/payment_attempts.csv \\
        --output data/payment_attempts_featured.csv

Then:
    python score_with_phase2_model.py \\
        --model-dir docs/phase2_plots \\
        --input data/payment_attempts_featured.csv \\
        --output data/payment_attempts_scored.csv

Then:
    python phase4_backtest.py --input data/payment_attempts_scored.csv

--------------------------------------------------------------------------
UPDATE: CLASS_ORDER and CATEGORICAL_COLUMNS below are now CONFIRMED against
train_classifier.py, not guessed. Two things were wrong before this pass:

1. CLASS_ORDER — confirmed to match train_classifier.py's DECLINE_REASONS
   exactly (same 8 categories, same order). decline_reason_classifier.txt
   is a raw Booster (native `.txt` save format, from `model.booster_.save_
   model()`), which has no `.classes_` — so this script always falls back
   to CLASS_ORDER_FALLBACK for this model, and that fallback is now
   verified correct rather than assumed.

2. CATEGORICAL_COLUMNS — was WRONG, not just unverified: the previous
   guess (`payment_method`, `gateway`, `bank_response_code` — 3 columns)
   is missing 7 of the 10 columns train_classifier.py actually passed as
   `categorical_feature` (`merchant_archetype`, `merchant_id`,
   `amount_bucket`, `hour_bucket`, `day_of_week`, `prev_decline_reason`,
   `prev_bank_response_code`). This is exactly the silent-failure mode
   this docstring used to warn about: with only 3 of 10 columns marked
   `category` dtype, LightGBM's positional remap of stored training-time
   category vocab (`booster.pandas_categorical`) onto this script's
   categorical-dtype columns would have been misaligned — wrong category
   codes on wrong columns, with no error raised. Fixed below.

3. REQUIRED_FEATURES — unchanged: still read directly off the loaded
   model (`feature_name()`), not hardcoded, so it can't drift out of sync
   with whatever feature set the model actually expects.

If train_classifier.py's CATEGORICAL_COLS or DECLINE_REASONS ever change,
re-verify these two constants against it again — they're now correct for
the training script as it exists today, not permanently guaranteed.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# CONFIRMED against train_classifier.py Cell 5's CATEGORICAL_COLS (10 columns,
# not the 3 previously guessed). Order doesn't matter here — prepare_features
# below marks these `category` dtype in whatever order the model's own
# feature_names list puts them in, which is what LightGBM actually keys off.
CATEGORICAL_COLUMNS = [
    "payment_method", "gateway", "bank_response_code", "merchant_archetype",
    "merchant_id", "amount_bucket", "hour_bucket", "day_of_week",
    "prev_decline_reason", "prev_bank_response_code",
]

# CONFIRMED against train_classifier.py Cell 1's DECLINE_REASONS — exact
# match, used as the class-index-to-name mapping whenever the loaded model
# is a raw Booster (no `.classes_`), which is the case for
# decline_reason_classifier.txt.
CLASS_ORDER_FALLBACK = [
    "insufficient_funds", "bank_server_error", "invalid_otp", "risk_block",
    "network_timeout", "card_expired", "limit_exceeded", "other",
]

# Rough test-set class proportions from phase2_results.md, used only as a
# sanity check on your scored output — NOT applied as a constraint or
# reweighting of any kind.
KNOWN_TEST_SET_PROPORTIONS = {
    "insufficient_funds": 310 / 1025, "bank_server_error": 227 / 1025,
    "invalid_otp": 124 / 1025, "risk_block": 84 / 1025,
    "network_timeout": 99 / 1025, "card_expired": 62 / 1025,
    "limit_exceeded": 68 / 1025, "other": 51 / 1025,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_model_file(model_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"--model-path {p} does not exist.")
        return p

    if not model_dir.exists():
        raise SystemExit(f"--model-dir {model_dir} does not exist.")

    candidates = []
    for ext in ("*.pkl", "*.joblib", "*.txt", "*.model", "*.lgb"):
        candidates.extend(model_dir.glob(ext))
    # a plain "model.txt" saved via booster.save_model() is the classic
    # LightGBM native-API artifact name; prefer it if present and unambiguous
    candidates = [c for c in candidates if c.is_file()]

    if not candidates:
        raise SystemExit(
            f"No model file found in {model_dir} (looked for *.pkl, *.joblib, "
            f"*.txt, *.model, *.lgb). Pass --model-path explicitly."
        )
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise SystemExit(
            f"Found multiple candidate files in {model_dir}: {names}. "
            f"Pass --model-path to disambiguate."
        )
    print(f"[score] using model file: {candidates[0]}", file=sys.stderr)
    return candidates[0]


class LoadedModel:
    """
    Thin wrapper unifying the two things a Phase 2 model might be:
      - a pickled/joblib'd sklearn-API model (lightgbm.LGBMClassifier),
        which carries .classes_, .feature_name_(), and .predict_proba()
      - a raw lightgbm.Booster (native API, e.g. booster.save_model(...)),
        which only carries .feature_name() and .predict() -> probabilities,
        with NO label names attached.
    """

    def __init__(self, model, kind: str):
        self.model = model
        self.kind = kind  # "sklearn" | "booster"

    @property
    def feature_names(self) -> List[str]:
        if self.kind == "sklearn":
            return list(self.model.feature_name_)
        return list(self.model.feature_name())

    @property
    def classes_from_model(self) -> Optional[List[str]]:
        if self.kind == "sklearn" and hasattr(self.model, "classes_"):
            return [str(c) for c in self.model.classes_]
        return None

    @property
    def pandas_categorical(self):
        booster = self.model.booster_ if self.kind == "sklearn" else self.model
        return getattr(booster, "pandas_categorical", None)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "sklearn":
            return self.model.predict_proba(X)
        # native Booster.predict() on a multiclass model already returns
        # (n_samples, n_classes) probabilities
        return self.model.predict(X)


def load_model(path: Path) -> LoadedModel:
    if path.suffix in (".pkl", ".joblib"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # sklearn wrapper: has predict_proba + classes_
        if hasattr(obj, "predict_proba") and hasattr(obj, "classes_"):
            return LoadedModel(obj, kind="sklearn")
        # someone pickled a bare Booster
        if hasattr(obj, "predict") and hasattr(obj, "feature_name"):
            return LoadedModel(obj, kind="booster")
        raise SystemExit(
            f"Loaded {path} but it's neither an sklearn-API classifier "
            f"(predict_proba + classes_) nor a lightgbm.Booster (predict + "
            f"feature_name). Got: {type(obj)}"
        )

    # .txt / .model / .lgb -> native Booster save format
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(path))
    return LoadedModel(booster, kind="booster")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_names: List[str],
                      categorical_columns: List[str]) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise SystemExit(
            f"Input CSV is missing {len(missing)} feature(s) the model expects: "
            f"{missing}. These must be present (already engineered) in the "
            f"input file — this script does not compute rolling/historical "
            f"features from raw event history."
        )

    X = df[feature_names].copy()
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def sanity_check_categorical_coverage(model: LoadedModel, categorical_columns: List[str]) -> None:
    pc = model.pandas_categorical
    if pc is None:
        return
    if len(pc) != len(categorical_columns):
        print(
            f"[score] WARNING: the loaded model recorded {len(pc)} categorical "
            f"column(s) at training time, but CATEGORICAL_COLUMNS here lists "
            f"{len(categorical_columns)} ({categorical_columns}). If these "
            f"counts don't match, category-code alignment between train and "
            f"predict will be wrong. Fix CATEGORICAL_COLUMNS at the top "
            f"of this file to match your actual training script.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(df: pd.DataFrame, model: LoadedModel, class_order_override: Optional[List[str]],
          categorical_columns: List[str]) -> pd.DataFrame:
    feature_names = model.feature_names
    print(f"[score] model expects {len(feature_names)} feature(s): {feature_names}", file=sys.stderr)

    X = prepare_features(df, feature_names, categorical_columns)
    sanity_check_categorical_coverage(model, categorical_columns)

    proba = model.predict_proba(X)  # (n_samples, n_classes)

    if class_order_override:
        class_order = class_order_override
        print(f"[score] using --class-order override: {class_order}", file=sys.stderr)
    elif model.classes_from_model:
        class_order = model.classes_from_model
        print(f"[score] using class order from model.classes_: {class_order}", file=sys.stderr)
    else:
        class_order = CLASS_ORDER_FALLBACK
        print(
            "[score] WARNING: model has no .classes_ (looks like a raw Booster) "
            "and no --class-order was given — falling back to the UNVERIFIED "
            f"guessed order: {class_order}. This is the single most likely "
            "silent-failure point in this script; confirm it against your "
            "training script's label encoder before trusting any output.",
            file=sys.stderr,
        )

    if proba.shape[1] != len(class_order):
        raise SystemExit(
            f"Model produced {proba.shape[1]} class probabilities but "
            f"class_order has {len(class_order)} entries ({class_order}). "
            f"These must match exactly — fix --class-order."
        )

    pred_idx = np.argmax(proba, axis=1)
    predicted_category = [class_order[i] for i in pred_idx]
    predicted_confidence = proba[np.arange(len(proba)), pred_idx]

    out = df.copy()
    out["predicted_category"] = predicted_category
    out["predicted_confidence"] = predicted_confidence
    return out


def print_sanity_check(out: pd.DataFrame) -> None:
    observed = out["predicted_category"].value_counts(normalize=True).to_dict()
    print("\n[score] predicted_category distribution vs. Phase 2's known test-set "
          "proportions (sanity check only — your batch need not match exactly, "
          "but a wildly different shape usually means class order or categorical "
          "encoding is misconfigured, not that the model got worse):", file=sys.stderr)
    for cat, known_pct in KNOWN_TEST_SET_PROPORTIONS.items():
        obs_pct = observed.get(cat, 0.0)
        print(f"    {cat:<20} known={known_pct*100:5.1f}%   this batch={obs_pct*100:5.1f}%", file=sys.stderr)
    unknown = set(observed) - set(KNOWN_TEST_SET_PROPORTIONS)
    if unknown:
        print(f"    WARNING: predicted categories not in the known taxonomy: {unknown} "
              f"— almost certainly a class-order mismatch.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score a payment-attempts CSV with the trained Phase 2 model.")
    ap.add_argument("--model-dir", type=str, default="docs/phase2_plots",
                     help="Directory to search for a single model file if --model-path isn't given.")
    ap.add_argument("--model-path", type=str, default=None,
                     help="Explicit path to the model file (.pkl/.joblib for a pickled sklearn classifier, "
                          ".txt/.model for a native lightgbm.Booster).")
    ap.add_argument("--input", type=str, required=True, help="Raw payment_attempts.csv to score.")
    ap.add_argument("--output", type=str, default=None,
                     help="Where to write the scored CSV. Defaults to <input>_scored.csv (never overwrites --input).")
    ap.add_argument("--class-order", type=str, default=None,
                     help="Comma-separated category names in the exact order the model's probability "
                          "columns come out in. Required if the model is a raw Booster (no .classes_).")
    ap.add_argument("--categorical-columns", type=str, default=",".join(CATEGORICAL_COLUMNS),
                     help="Comma-separated columns to encode as pandas 'category' dtype before predicting.")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"--input {input_path} does not exist.")
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_scored.csv")
    if output_path.resolve() == input_path.resolve():
        raise SystemExit("--output must not be the same file as --input.")

    model_path = find_model_file(Path(args.model_dir), args.model_path)
    model = load_model(model_path)

    df = pd.read_csv(input_path)
    print(f"[score] loaded {len(df):,} rows from {input_path}", file=sys.stderr)

    class_order_override = args.class_order.split(",") if args.class_order else None
    categorical_columns = [c.strip() for c in args.categorical_columns.split(",") if c.strip()]

    out = score(df, model, class_order_override, categorical_columns)
    print_sanity_check(out)

    out.to_csv(output_path, index=False)
    print(f"\n[score] wrote {output_path}", file=sys.stderr)
    print(f"[score] next: python phase4_backtest.py --input {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
