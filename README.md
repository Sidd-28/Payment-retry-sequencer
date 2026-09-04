# Payment Retry Sequencer

**Track:** AI Revenue Recovery (Razorpay Buildathon)

Detects *why* a payment attempt failed and recommends the optimal retry action — or correctly recommends no retry — to recover legitimate revenue without ever bypassing fraud or risk controls.

> Status: 🚧 Phase 0 complete (schema + repo skeleton). See [build plan](./docs/) for phase-by-phase progress.

---

## What it solves

Payment failures aren't one clean event — a card times out, a mandate bounces, an OTP expires. Most of those are recoverable if you react correctly and quickly; some (fraud blocks, expired cards) are not and should never be retried. This project builds the decision layer that tells the difference, then measures the actual revenue impact of getting it right.

---

## Architecture

```
[Data Simulator] → [Failure Classifier] → [Retry Policy Engine] → [Backtest/Eval Harness]
                                                    ↓
                                          [Serving API] → [Dashboard]
                                                    ↓
                                    [Audit Trail / Guardrails] (cross-cutting)
```

*(Full component breakdown to be added as each phase is built — see `/docs`.)*

---

## Repo structure

```
payment-retry-sequencer/
├── data/               synthetic datasets
├── notebooks/          exploration + model training notebooks
├── docs/                schema.json, schema.md, and phase reports
├── src/
│   ├── classifier/      Phase 2 — failure root-cause classifier
│   ├── policy/          Phase 3 — retry policy engine + hard guardrails
│   ├── backtest/         Phase 4 — money-recovered evaluation harness
│   ├── api/              Phase 5 — FastAPI serving layer
│   ├── dashboard/        Phase 6 — Streamlit dashboard
│   └── audit/             Phase 7 — immutable audit log + guardrail enforcement
└── README.md
```

---

## Setup

```bash
# TODO once Phase 1+ dependencies are added
pip install -r requirements.txt
```

---

## Results

*(To be filled in after Phase 4 — headline numbers: revenue recovered vs. baseline, false-positive rate, guardrail-compliance rate.)*

---

## What broke (and how I got out)

*(To be filled in honestly during development — Phase 8.)*

---

## Assumptions & limitations

- Uses fully synthetic transaction data; no real merchant or customer data.
- Decline-rate distributions are calibrated against publicly available statistics, not Razorpay's actual data.
- `risk_block` and `card_expired` are always hard-stopped from retry — this is a fixed rule, not a tunable model parameter.
