# %% [markdown]
# # Payment Retry Sequencer — Phase 2: Decline-Reason Classifier
#
# Re-derives the true `decline_reason_category` from noisy, real-world-style
# signals (`bank_response_code` + context), simulating the case where the raw
# bank code alone isn't reliable enough to act on.
#
# Cells are separated with `# %%` (works as-is in VS Code / Jupytext / Spyder;
# paste into a Jupyter notebook cell-by-cell if you prefer).
#
# **Hard rule reminder (do not relearn this in the model):** `risk_block` and
# `card_expired` must map to `no_retry` regardless of what this classifier
# predicts elsewhere in the pipeline. This script's job is only to get the
# *category* right as often as possible — Phase 3 is responsible for making
# sure the no-retry rule for those two categories is enforced outside the
# model, so a bad retrain can't silently erode it.

# %%
# --- Cell 1: Imports & config ------------------------------------------------
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

matplotlib.use("Agg")  # headless; swap to an interactive backend in a real notebook
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "data/payment_attempts.csv"        # Phase 1 output
OUTPUT_DIR = Path("./phase2_output")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# The full, fixed taxonomy — order matters, this defines the label encoding.
DECLINE_REASONS = [
    "insufficient_funds", "bank_server_error", "invalid_otp", "risk_block",
    "network_timeout", "card_expired", "limit_exceeded", "other",
]
# Categories whose ideal action is anything OTHER than a hard stop.
# If a true risk_block / card_expired gets predicted into one of these,
# a downstream policy engine could act on it as if it were retryable —
# that's the dangerous failure mode this script explicitly measures later.
RETRYABLE_REASONS = {
    "insufficient_funds", "bank_server_error", "invalid_otp",
    "network_timeout", "limit_exceeded",
}
NO_RETRY_REASONS = {"risk_block", "card_expired", "other"}

# Columns that must NEVER be used as model input features.
BANNED_PREFIXES = ("gt_", "ctx_")

# %%
# --- Cell 2: Load data & sanity checks ---------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Rows total: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
print(f"Failed attempts (rows with a label): {(~df['success']).sum():,}")

banned_present = [c for c in df.columns if c.startswith(BANNED_PREFIXES)]
print(f"\nBanned (gt_/ctx_) columns present in raw data (will be dropped before featurizing): {banned_present}")

print("\ndecline_reason_category counts (failed attempts only):")
print(df.loc[~df["success"], "decline_reason_category"].value_counts())

# %%
# --- Cell 3: Feature engineering ---------------------------------------------
# All features below are computable from information a real system would
# actually have at prediction time: the raw event fields, the timestamp, and
# each transaction's OWN prior-attempt history (`previous_attempts`).
#
# Two things are deliberately NOT done here, on purpose:
#   1. gt_* / ctx_* columns are dropped outright (see BANNED_PREFIXES) —
#      they encode either the label itself or the synthetic generator's
#      internal simulation state (e.g. "is this gateway degraded today"),
#      which a real classifier would never have direct access to.
#   2. Gateway maintenance windows are NOT hardcoded as a feature, even
#      though Phase 1's generator config defines exact maintenance hours
#      per gateway. Encoding that here would mean peeking at the data
#      generator's internal parameters rather than learning from observed
#      behavior. Instead, `gateway_recent_failure_rate` (a rolling, purely
#      historical signal) lets the model discover gateway degradation
#      empirically — the same way a production system would have to.

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
    (successes included) BEFORE splitting, then the split just slices rows —
    this is what makes it leakage-safe: each row only ever sees the past."""
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

    # Gateway health: a SHORT trailing window, not full expanding history —
    # gateway reliability drifts day to day, so "last ~300 attempts" is a much
    # more honest proxy for "is this gateway currently degraded" than an
    # all-time average would be.
    frame["gateway_recent_failure_rate"] = (
        frame.groupby("gateway")["failed"]
        .transform(lambda s: s.rolling(window=300, min_periods=10).mean().shift(1))
    )
    return frame


df = add_time_features(df)
df = add_previous_attempt_features(df)
df = add_rolling_history_features(df)  # run on FULL frame, incl. successes, before any split

print("New columns added:")
print([c for c in df.columns if c not in pd.read_csv(DATA_PATH, nrows=1).columns])

# %%
# --- Cell 4: Time-based split -------------------------------------------------
# Cutoffs are quantiles of the FULL (success+fail) time-sorted frame, so the
# split reflects calendar time, not row-count-of-failures. This is the whole
# point of a time split: the model must never see anything about "the future"
# relative to what it's evaluated on, and rolling features must not be
# recomputed per-split (they're already correct — see Cell 3's docstring).
train_cut = df["timestamp"].quantile(0.70)
val_cut = df["timestamp"].quantile(0.85)
print(f"train < {train_cut}  |  val < {val_cut}  |  test >= {val_cut}")

df["split"] = np.select(
    [df["timestamp"] < train_cut, df["timestamp"] < val_cut],
    ["train", "val"],
    default="test",
)
print(df["split"].value_counts())

# Amount bucket edges are fit on TRAIN only, then applied everywhere —
# fitting them on the full dataset would leak the val/test amount
# distribution into a feature used at train time.
train_amounts = df.loc[df["split"] == "train", "amount"]
_, bin_edges = pd.qcut(train_amounts, q=5, retbins=True, duplicates="drop")
bin_edges = bin_edges.copy()
bin_edges[0], bin_edges[-1] = -np.inf, np.inf
bucket_labels = [f"q{i+1}" for i in range(len(bin_edges) - 1)]
df["amount_bucket"] = pd.cut(df["amount"], bins=bin_edges, labels=bucket_labels)

# Now restrict to rows that actually have a label to predict.
labeled = df[~df["success"]].copy()

# Cold-start fill value for merchant/customer rolling rates: the TRAIN-period
# global failure rate (a leak-safe constant, not derived from val/test).
train_global_fail_rate = df.loc[df["split"] == "train", "failed"].mean()
for col in ["merchant_hist_failure_rate", "customer_hist_failure_rate", "gateway_recent_failure_rate"]:
    labeled[col] = labeled[col].fillna(train_global_fail_rate)

print(f"\nLabeled (failed) rows: {len(labeled):,}")
print(labeled["split"].value_counts())

# %%
# --- Cell 5: Assemble X / y ---------------------------------------------------
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
CATEGORICAL_COLS = [
    "payment_method", "gateway", "bank_response_code", "merchant_archetype", "merchant_id",
    "amount_bucket", "hour_bucket", "day_of_week", "prev_decline_reason", "prev_bank_response_code",
]

# Explicitly NOT included, and why:
#   transaction_id / payment_session_id / customer_id (raw) — unique-ish IDs;
#       customer_id's generalizable signal is already captured by
#       customer_hist_failure_rate + customer_prior_attempt_count.
#   currency          — constant ("INR"), zero information.
#   success            — constant (False) within `labeled`, by construction.
#   previous_attempts (raw JSON) — already mined into the prev_* features.
#   retry_action_taken / retry_outcome / time_to_recovery — these are Phase 3
#       POLICY ENGINE outputs. They're null in this dataset, but structurally
#       they're downstream of the very prediction this model makes — using
#       them as inputs would be circular, even on a version of the data where
#       they're populated.
#   any gt_* / ctx_* column — banned by design (see Cell 3).

assert not any(c.startswith(BANNED_PREFIXES) for c in FEATURE_COLS), "Banned column leaked into features!"

labeled["decline_reason_category"] = pd.Categorical(
    labeled["decline_reason_category"], categories=DECLINE_REASONS
)
labeled["label"] = labeled["decline_reason_category"].cat.codes

for c in CATEGORICAL_COLS:
    # Fit the category VOCABULARY on the full dataset (just "what values can
    # this column take", not a statistic) so train/val/test share consistent
    # codes. This is not leakage — it's the same reasoning as knowing the
    # fixed 8-value decline-reason taxonomy up front.
    labeled[c] = labeled[c].astype("category")

X_train = labeled.loc[labeled["split"] == "train", FEATURE_COLS]
X_val = labeled.loc[labeled["split"] == "val", FEATURE_COLS]
X_test = labeled.loc[labeled["split"] == "test", FEATURE_COLS]
y_train = labeled.loc[labeled["split"] == "train", "label"]
y_val = labeled.loc[labeled["split"] == "val", "label"]
y_test = labeled.loc[labeled["split"] == "test", "label"]

print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

# %%
# --- Cell 6: Class-imbalance handling -----------------------------------------
# LightGBM's sklearn `class_weight` param is reliable for binary problems but
# not multiclass; the portable, sklearn-native approach is to compute
# per-sample weights with `compute_sample_weight(class_weight="balanced")`
# and pass them to `.fit(sample_weight=...)`. This works identically with
# XGBoost if you swap the estimator.
sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
sample_weight_val = compute_sample_weight(class_weight="balanced", y=y_val)

# On top of balanced weighting, apply an extra recall-biasing multiplier to
# risk_block specifically. This is a deliberate, asymmetric cost choice: a
# false NEGATIVE here (missing a real risk_block) is far more dangerous than
# a false positive (over-flagging something as risk_block, which just costs
# a manual review). Tune RISK_BLOCK_SAFETY_MULTIPLIER against your own
# business cost estimates — this is not free; pushing it up will trade away
# some risk_block precision for recall.
RISK_BLOCK_SAFETY_MULTIPLIER = 1.5
risk_block_code = DECLINE_REASONS.index("risk_block")
sample_weight_train = np.where(
    y_train.values == risk_block_code, sample_weight_train * RISK_BLOCK_SAFETY_MULTIPLIER, sample_weight_train
)

print("Class weight range (train):", sample_weight_train.min(), "-", sample_weight_train.max())

# %%
# --- Cell 7: Train LightGBM multiclass classifier -----------------------------
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(DECLINE_REASONS),
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[sample_weight_val],
    eval_metric="multi_logloss",
    categorical_feature=CATEGORICAL_COLS,
    callbacks=[lgb.early_stopping(stopping_rounds=75), lgb.log_evaluation(period=100)],
)

print(f"\nBest iteration: {model.best_iteration_}")

# %%
# --- Cell 8: Evaluate on the held-out TEST split -------------------------------
y_pred = model.predict(X_test)
y_pred_labels = [DECLINE_REASONS[i] for i in y_pred]
y_test_labels = [DECLINE_REASONS[i] for i in y_test]

print(classification_report(y_test_labels, y_pred_labels, labels=DECLINE_REASONS, digits=3, zero_division=0))
macro_f1 = f1_score(y_test_labels, y_pred_labels, labels=DECLINE_REASONS, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test_labels, y_pred_labels, labels=DECLINE_REASONS, average="weighted", zero_division=0)
print(f"Macro F1: {macro_f1:.3f}   Weighted F1: {weighted_f1:.3f}")

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=DECLINE_REASONS)
cm_df = pd.DataFrame(cm, index=DECLINE_REASONS, columns=DECLINE_REASONS)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
print("\nConfusion matrix (rows=true, cols=predicted):")
print(cm_df)

fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DECLINE_REASONS)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=True, values_format="d")
plt.title("Decline-reason classifier — test set confusion matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# %%
# --- Cell 9: Dangerous-failure-mode analysis -----------------------------------
# The failure mode this defensive system cares about most: a TRUE risk_block
# (or card_expired) getting predicted as something in RETRYABLE_REASONS. If
# that happens, and the downstream hard rule keys off the PREDICTED category
# (rather than some independent fraud signal), a blocked/fraud transaction
# could get a retry action applied to it.

def dangerous_misclassification_report(true_labels, pred_labels, hard_stop_category):
    mask = np.array(true_labels) == hard_stop_category
    n = mask.sum()
    if n == 0:
        print(f"No {hard_stop_category} examples in test set.")
        return
    preds_for_true = np.array(pred_labels)[mask]
    into_retryable = np.isin(preds_for_true, list(RETRYABLE_REASONS)).sum()
    into_other_no_retry = np.isin(preds_for_true, list(NO_RETRY_REASONS - {hard_stop_category})).sum()
    correct = (preds_for_true == hard_stop_category).sum()
    print(f"\nTrue label = {hard_stop_category}  (n={n} in test set)")
    print(f"  Correctly predicted {hard_stop_category}:               {correct:4d}  ({correct/n:.1%})")
    print(f"  Predicted as ANOTHER no_retry category ({sorted(NO_RETRY_REASONS - {hard_stop_category})}): "
          f"{n - correct - into_retryable:4d}  ({(n - correct - into_retryable)/n:.1%})  <- action still safe (no_retry either way)")
    print(f"  ** Predicted as a RETRYABLE category (DANGEROUS): {into_retryable:4d}  ({into_retryable/n:.1%}) **")
    if into_retryable > 0:
        wrong_dist = pd.Series(preds_for_true[np.isin(preds_for_true, list(RETRYABLE_REASONS))]).value_counts()
        print(f"    Breakdown of dangerous mispredictions:\n{wrong_dist.to_string()}")


print("=" * 70)
print("DANGEROUS FAILURE MODE CHECK")
print("=" * 70)
dangerous_misclassification_report(y_test_labels, y_pred_labels, "risk_block")
dangerous_misclassification_report(y_test_labels, y_pred_labels, "card_expired")

# %%
# --- Cell 10: SHAP explainability ----------------------------------------------
explainer = shap.TreeExplainer(model)
# Use a manageable background/explain sample for speed.
X_explain = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer.shap_values(X_explain)  # multiclass -> list/array per class

# Global feature importance, per class, for a quick sanity check.
if isinstance(shap_values, list):
    mean_abs_by_class = {
        DECLINE_REASONS[i]: np.abs(shap_values[i]).mean(axis=0) for i in range(len(DECLINE_REASONS))
    }
else:
    # newer shap returns array shaped (n_samples, n_features, n_classes)
    mean_abs_by_class = {
        DECLINE_REASONS[i]: np.abs(shap_values[:, :, i]).mean(axis=0) for i in range(len(DECLINE_REASONS))
    }
importance_df = pd.DataFrame(mean_abs_by_class, index=FEATURE_COLS)
importance_df["overall"] = importance_df.mean(axis=1)
importance_df = importance_df.sort_values("overall", ascending=False)
importance_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv")
print("Top 10 features by mean |SHAP| (averaged across classes):")
print(importance_df["overall"].head(10).to_string())

# --- 3 example predictions, explained ---
# 1) A correctly predicted risk_block (the safety-critical case)
# 2) A correctly predicted insufficient_funds (the highest-volume case)
# 3) Whatever the model got MOST wrong, by predicted-probability margin
#    (useful for seeing what confuses it)
proba_test = model.predict_proba(X_test)
test_idx = X_test.index

examples = {}
risk_block_correct = [
    i for i, (t, p) in enumerate(zip(y_test_labels, y_pred_labels))
    if t == "risk_block" and p == "risk_block"
]
if risk_block_correct:
    examples["risk_block_correct"] = risk_block_correct[0]

insuff_correct = [
    i for i, (t, p) in enumerate(zip(y_test_labels, y_pred_labels))
    if t == "insufficient_funds" and p == "insufficient_funds"
]
if insuff_correct:
    examples["insufficient_funds_correct"] = insuff_correct[0]

wrong_idxs = [i for i, (t, p) in enumerate(zip(y_test_labels, y_pred_labels)) if t != p]
if wrong_idxs:
    # pick the misclassification the model was most CONFIDENT about — the most instructive kind of mistake
    confidences = [proba_test[i].max() for i in wrong_idxs]
    examples["most_confident_mistake"] = wrong_idxs[int(np.argmax(confidences))]

print(f"\nExplaining {len(examples)} example predictions -> saved as PNGs in {OUTPUT_DIR}/")
for name, i in examples.items():
    row_index_in_test = i
    true_lbl = y_test_labels[row_index_in_test]
    pred_lbl = y_pred_labels[row_index_in_test]
    pred_class_idx = DECLINE_REASONS.index(pred_lbl)
    x_row = X_test.iloc[[row_index_in_test]]

    row_shap = explainer.shap_values(x_row)
    if isinstance(row_shap, list):
        vals = row_shap[pred_class_idx][0]
        base = explainer.expected_value[pred_class_idx]
    else:
        vals = row_shap[0, :, pred_class_idx]
        base = explainer.expected_value[pred_class_idx]

    print(f"\n[{name}]  true={true_lbl}  predicted={pred_lbl}  "
          f"P(predicted)={proba_test[row_index_in_test][pred_class_idx]:.2f}")
    top_features = pd.Series(vals, index=FEATURE_COLS).sort_values(key=np.abs, ascending=False).head(6)
    for feat, val in top_features.items():
        print(f"    {feat:32s} value={x_row.iloc[0][feat]!s:15s} shap={val:+.3f}")

    fig = plt.figure(figsize=(9, 4))
    expl = shap.Explanation(values=vals, base_values=base, data=x_row.iloc[0].values, feature_names=FEATURE_COLS)
    shap.plots.waterfall(expl, show=False, max_display=10)
    plt.title(f"{name}: true={true_lbl}, predicted={pred_lbl}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

# %%
# --- Cell 11: Save model & artifacts --------------------------------------------
model.booster_.save_model(str(OUTPUT_DIR / "decline_reason_classifier.txt"))
with open(OUTPUT_DIR / "feature_columns.json", "w") as f:
    json.dump({"features": FEATURE_COLS, "categorical": CATEGORICAL_COLS, "classes": DECLINE_REASONS}, f, indent=2)

results = {
    "test_macro_f1": macro_f1,
    "test_weighted_f1": weighted_f1,
    "best_iteration": int(model.best_iteration_),
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved model + artifacts to {OUTPUT_DIR.resolve()}")
print(json.dumps(results, indent=2))
