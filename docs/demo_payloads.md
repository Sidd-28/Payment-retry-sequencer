# Demo payloads — used in the pitch video

These are the exact request bodies sent to `POST /evaluate-attempt`
(`localhost:8000/docs`) in the pitch video. Paste either one directly into
the Swagger UI's "Try it out" box to reproduce the results shown in the
demo.

## Demo 1 — insufficient_funds, ML-scored retry

Predicted category: `insufficient_funds` (97.77% confidence)
Recommended action: `alt_payment_method_prompt`, immediate

```json
{
  "transaction_id": "txn_demo_001",
  "merchant_id": "merch_0042",
  "customer_id": "cust_00981",
  "amount": 499,
  "currency": "INR",
  "payment_method": "upi",
  "gateway": "gateway_b",
  "bank_response_code": "U69",
  "timestamp": "2026-09-05T14:32:00+05:30",
  "attempt_number": 1,
  "is_recurring": true,
  "previous_attempts": [],
  "merchant_hist_failure_rate": 0.15,
  "merchant_prior_attempt_count": 5000,
  "customer_hist_failure_rate": 0.05,
  "customer_prior_attempt_count": 3,
  "gateway_recent_failure_rate": 0.10
}
```

## Demo 2 — risk_block, hard-rule no_retry

Predicted category: `risk_block` (98.24% confidence)
Recommended action: `no_retry` (hard rule, regardless of classifier
confidence — flagged for mandatory human review, no alternate
gateway/method suggested)

```json
{
  "transaction_id": "txn_demo_002",
  "merchant_id": "merch_0117",
  "customer_id": "cust_04456",
  "amount": 49900,
  "currency": "INR",
  "payment_method": "card",
  "gateway": "gateway_a",
  "bank_response_code": "05",
  "timestamp": "2026-09-05T02:14:00+05:30",
  "attempt_number": 4,
  "is_recurring": false,
  "previous_attempts": [
    { "attempt_number": 1, "timestamp": "2026-09-05T02:01:00+05:30", "bank_response_code": "05", "decline_reason_category": "risk_block", "retry_outcome": "failed" },
    { "attempt_number": 2, "timestamp": "2026-09-05T02:06:00+05:30", "bank_response_code": "12", "decline_reason_category": "other", "retry_outcome": "failed" },
    { "attempt_number": 3, "timestamp": "2026-09-05T02:10:00+05:30", "bank_response_code": "05", "decline_reason_category": "risk_block", "retry_outcome": "failed" }
  ],
  "merchant_hist_failure_rate": 0.15,
  "merchant_prior_attempt_count": 5000,
  "customer_hist_failure_rate": 0.05,
  "customer_prior_attempt_count": 3,
  "gateway_recent_failure_rate": 0.10
}
```

## Note

`bank_response_code: "05"` is a genuine risk-coded value from the Phase 1
data generator's card response-code pool (`07, 41, 59, 05`), not a
placeholder. `bank_response_code: "U69"` is a real UPI insufficient-funds
code from the same generator. Both responses shown above were verified
live against the running API, not assumed.
