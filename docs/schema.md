# Data Schema — Payment Attempt Event

This is the canonical record shape used across every phase of the project (data generator, classifier, policy engine, backtest, API). The machine-readable version lives in [`schema.json`](./schema.json).

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `transaction_id` | string | ✅ | Unique ID for this specific attempt. |
| `merchant_id` | string | ✅ | Synthetic merchant identifier. |
| `customer_id` | string | ✅ | Hashed/synthetic customer identifier — never real PII. |
| `amount` | number | ✅ | Transaction amount, whole rupees. |
| `currency` | string | ✅ | Fixed to `"INR"` for this project's scope. |
| `payment_method` | enum | ✅ | `card` \| `upi` \| `netbanking` \| `wallet` |
| `gateway` | string | ✅ | Synthetic gateway name (e.g. `gateway_a`), not a real vendor name. |
| `bank_response_code` | string | ✅ | Raw, noisy code as if received from the bank — the signal the classifier has to interpret. |
| `decline_reason_category` | enum | ✅ | The **true** underlying reason (ground truth in synthetic data). See taxonomy below. |
| `timestamp` | ISO-8601 datetime | ✅ | When the attempt occurred. |
| `attempt_number` | integer ≥ 1 | ✅ | 1 for the original attempt, incrementing per retry. |
| `is_recurring` | boolean | ✅ | True for subscription/mandate payments. |
| `previous_attempts` | array | — | History of prior attempts for this transaction, oldest first. |
| `retry_action_taken` | enum \| null | — | What the policy engine decided. Null until evaluated. |
| `retry_outcome` | enum \| null | — | `recovered` \| `failed` \| `pending` \| null. |
| `time_to_recovery` | number \| null | — | Seconds from failure to recovery. Null if never recovered. |

## `decline_reason_category` taxonomy

| Value | Meaning | Retryable? |
|---|---|---|
| `insufficient_funds` | Account/card lacked funds at time of attempt | Yes — but only after a delay (e.g. near typical salary dates) |
| `bank_server_error` | Transient failure on the bank/gateway side | Yes — usually immediately or after a short delay |
| `invalid_otp` | Customer entered OTP incorrectly or it expired | Yes — via a fresh customer-initiated attempt, not silent auto-retry |
| `risk_block` | Transaction blocked by fraud/risk controls | **No — hard stop, human review required** |
| `network_timeout` | Request never completed due to network issues | Yes — immediately retryable, often same gateway |
| `card_expired` | Card details are stale | **No — requires new card details, not a retry** |
| `limit_exceeded` | Per-transaction or daily limit hit | Yes — via alternate payment method, not same-method retry |
| `other` | Anything not confidently classified | No automated action — flag for review |

## Retry-action taxonomy

| Action | When it applies | Notes |
|---|---|---|
| `same_gateway_retry` | Transient errors: `network_timeout`, `bank_server_error` | Simplest, lowest cost |
| `alt_gateway_retry` | Gateway-specific degradation suspected | Requires gateway health signal |
| `alt_payment_method_prompt` | `limit_exceeded`, repeated same-method failures | Customer-facing prompt, not silent |
| `delayed_retry` | `insufficient_funds` | Carries a `timing_hint` (e.g. "retry after 3 days") |
| `no_retry` | `risk_block`, `card_expired`, `other` | **Hard stop — enforced outside the ML model, see Phase 3/7** |

## Design notes

- `decline_reason_category` is generated as ground truth in the synthetic dataset, then the classifier (Phase 2) is trained to *re-derive* it from noisier features — simulating the real-world case where you only have a raw, unreliable bank response code.
- `risk_block` and `card_expired` map to `no_retry` by a **hard rule**, not a learned preference. This rule must live outside the ML model so it can't be silently changed by retraining (see Phase 3 and Phase 7 in the build plan).
