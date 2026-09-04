# Phase 2 — Decline-Reason Classifier: Results Summary

LightGBM multiclass classifier, trained to re-derive the true `decline_reason_category`
from noisy raw signals (`bank_response_code` + engineered context), on a **time-based
split** of the Phase 1 synthetic dataset (train: first 70% of calendar time, val: next
15%, test: final 15% — 4,064 / 914 / 1,025 labeled rows respectively). Numbers below are
from an actual run on the default Phase 1 dataset (seed 42, ~63k total attempt rows,
6,003 failed/labeled).

## Headline numbers

- **Test macro F1: 0.872** · **Test weighted F1: 0.878** · **Overall accuracy: 87.7%**
- Best iteration 131/2000 (early-stopped on validation `multi_logloss`)

## Per-class performance (test set)

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| insufficient_funds | 0.979 | 0.919 | 0.948 | 310 |
| bank_server_error | 0.854 | 0.802 | 0.827 | 227 |
| invalid_otp | 0.776 | 0.839 | 0.806 | 124 |
| risk_block | 0.862 | 0.893 | 0.877 | 84 |
| network_timeout | 0.824 | 0.848 | 0.836 | 99 |
| card_expired | 0.937 | 0.952 | 0.944 | 62 |
| limit_exceeded | 0.816 | 0.912 | 0.861 | 68 |
| other | 0.814 | 0.941 | 0.873 | 51 |

`insufficient_funds` and `card_expired` are the easiest to identify (high, distinctive
signal in `bank_response_code`). `invalid_otp` is the hardest — it's most often confused
with `bank_server_error` (11 of 124 true `invalid_otp` cases), which tracks: both can
present as a generic-looking soft decline before the raw code is disambiguated.

## Confusion matrix (test set, rows = true, columns = predicted)

| true \ pred | insuff._funds | bank_srv_err | invalid_otp | risk_block | net_timeout | card_exp. | limit_exc. | other |
|---|---|---|---|---|---|---|---|---|
| **insufficient_funds** | 285 | 5 | 6 | 4 | 2 | 0 | 4 | 4 |
| **bank_server_error** | 4 | 182 | 20 | 3 | 13 | 0 | 4 | 1 |
| **invalid_otp** | 0 | 11 | 104 | 2 | 1 | 0 | 5 | 1 |
| **risk_block** | 0 | 1 | 0 | 75 | 1 | 2 | 1 | 4 |
| **network_timeout** | 1 | 12 | 1 | 1 | 84 | 0 | 0 | 0 |
| **card_expired** | 0 | 0 | 1 | 1 | 1 | 59 | 0 | 0 |
| **limit_exceeded** | 1 | 0 | 2 | 1 | 0 | 1 | 62 | 1 |
| **other** | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 48 |

(Full matrix also saved as `confusion_matrix.csv` / `.png`.)

## The safety-critical number: dangerous misclassification rate

This is the metric that actually matters for a defensive system — a `risk_block` or
`card_expired` case getting predicted as something a downstream policy engine might
treat as retryable:

| True category | n (test) | Correctly identified | Misrouted to another *no_retry* category (still safe) | **Misrouted to a RETRYABLE category (dangerous)** |
|---|---|---|---|---|
| `risk_block` | 84 | 75 (89.3%) | 6 (7.1%) | **3 (3.6%)** |
| `card_expired` | 62 | 59 (95.2%) | 1 (1.6%) | **2 (3.2%)** |

The 3 dangerous `risk_block` misses went to `network_timeout` (1), `bank_server_error`
(1), and `limit_exceeded` (1). The 2 dangerous `card_expired` misses went to
`network_timeout` (1) and `invalid_otp` (1).

**This is not zero, and it shouldn't be presented as such.** ~3.6% of true fraud/risk
holds in this test set would, if this classifier's raw output were trusted blindly,
get routed toward a retry action. This is exactly why the project's design notes
require the `risk_block`/`card_expired` → `no_retry` mapping to live as a hard rule
**outside** the model (Phase 3), rather than trusting the classifier's category call
alone to gate retry eligibility — e.g., cross-checking against the raw
`bank_response_code`'s own risk-coded values, or requiring a second, independent
signal before ever authorizing a retry. A macro-F1 of 0.87 is a good classifier by ML
standards and still not good enough to be the only thing standing between a blocked
transaction and a retry.

`risk_block` recall (89.3%) benefited from the extra `RISK_BLOCK_SAFETY_MULTIPLIER =
1.5` sample-weighting applied during training — an explicit, documented choice to
trade some `risk_block` precision for recall, since a missed block is more costly than
an over-flagged one. Precision on `risk_block` (86.2%) is correspondingly a bit lower
than it would be under plain balanced weighting; retune the multiplier against real
cost estimates before relying on this in production.

## What actually drives the predictions (SHAP)

Across all 8 classes, `bank_response_code` dominates mean |SHAP| by roughly a 4x margin
over the next feature (`merchant_prior_attempt_count`), which is exactly what you'd
want: the raw code is genuinely informative, and the model isn't leaning on the
engineered context features as a crutch. Full ranking in `shap_feature_importance.csv`.

Top 10 features by mean |SHAP| (averaged across all 8 classes):

1. `bank_response_code` — 1.070
2. `merchant_prior_attempt_count` — 0.248
3. `day_of_month` — 0.103
4. `gateway_recent_failure_rate` — 0.097
5. `merchant_hist_failure_rate` — 0.090
6. `payment_method` — 0.089
7. `amount` — 0.088
8. `hour` — 0.067
9. `customer_prior_attempt_count` — 0.054
10. `customer_hist_failure_rate` — 0.036

## Three explained example predictions

1. **`risk_block_correct`** (`shap_risk_block_correct.png`) — true and predicted both
   `risk_block`, P=0.90. `bank_response_code = U39` alone contributes +4.17 (dominant);
   a slightly-below-average `merchant_hist_failure_rate` pulls very slightly the other
   way, everything else is minor.
2. **`insufficient_funds_correct`** (`shap_insufficient_funds_correct.png`) — true and
   predicted both `insufficient_funds`, P=0.96. Again `bank_response_code = U69` carries
   the prediction (+3.44); `gateway_recent_failure_rate` at an elevated 9.3% nudges it
   further in the same direction.
3. **`most_confident_mistake`** (`shap_most_confident_mistake.png`) — true
   `insufficient_funds`, predicted `limit_exceeded` at P=0.98. `bank_response_code =
   U51` drives this almost entirely (+4.48) — and `U51` is literally the code pool
   Phase 1 assigns to `limit_exceeded` under UPI. This row is one of the ~8% where
   Phase 1's generator deliberately sampled a raw code from the *wrong* category's pool
   to simulate signal noise. In other words: **the model didn't make a reasoning
   error here — it correctly read a code that was, by construction, misleading.** This
   is a useful example to keep in the README verbatim: it's the clearest illustration
   of why `bank_response_code` alone isn't trustworthy and why this whole phase exists.

## Caveats to carry into Phase 3/4

- Test-set support for the rarest classes (`card_expired` n=62, `other` n=51,
  `limit_exceeded` n=68) is thin. Precision/recall estimates for these categories have
  real sampling noise — a handful of examples flipping would move these numbers by
  several points. Scale up Phase 1's `daily_volume_range` before treating these
  per-class numbers as stable.
- `customer_id` was deliberately **excluded** as a raw categorical feature (12,501
  distinct values, most with very little history) — its generalizable signal is
  captured instead via `customer_hist_failure_rate` and `customer_prior_attempt_count`.
  If you have a real dataset where certain customers are consistently high-risk in a
  way rolling stats don't fully capture, revisit this.
- Gateway maintenance windows were **not** hardcoded as a feature, on purpose — that
  would mean leaking the synthetic generator's internal config rather than learning
  gateway degradation from observed behavior. `gateway_recent_failure_rate` (a 300-
  transaction rolling window) stands in for it, and shows up as the #4 most important
  feature, suggesting the model is in fact picking up on gateway health.
- All `gt_*`/`ctx_*` columns were dropped before feature construction (asserted in
  code, not just by convention) and `retry_action_taken`/`retry_outcome`/
  `time_to_recovery` were excluded on structural grounds (they're Phase 3 outputs,
  not Phase 2 inputs) even though they happen to be null in this dataset.
