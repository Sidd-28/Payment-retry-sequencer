# Payment Retry Sequencer — Phase 4 Backtest Report

Run: 2026-09-04T11:44:58.262327+00:00
Input: data/payment_attempts_scored.csv
Seed: 42
Transactions: 6,003 (total at-risk amount: ₹10,807,047.00)

> **Read before trusting the dollar figures:** this is a decision-logic
> backtest. Recovered-revenue figures come from a documented, seeded
> simulation of recovery outcomes (`build_true_recovery_probs`), not from
> observed retry outcomes — see the script docstring. The false-positive
> and guardrail-compliance numbers below are different: they compare the
> policy's actions against ground-truth labels, independent of that
> simulation.

## Headline: Net Recovered Revenue

Both figures below are shown side by side because they answer different questions: **gross (raw)** is total revenue recovered with no costs subtracted; **net (cost-adjusted)** subtracts gateway fees on every retry attempted plus a customer-friction/annoyance cost on every failed retry — including the false-positive retries in section 1 below (see that section for exactly how much of the deduction they account for).

| | Gross (raw) recovered | Net (cost-adjusted) recovered | % of at-risk (net) | Retries attempted | Recovered n |
|---|---|---|---|---|---|
| **Policy engine** | ₹5,353,850.00 | ₹5,334,298.00 | 49.36% | 4,936 | 3,000 |
| Baseline A (no retry ever) | ₹0.00 | ₹0.00 | 0.00% | 0 | 0 |
| Baseline B (naive retry-once) | ₹2,827,060.00 | ₹2,793,089.00 | 25.85% | 6,003 | 1,610 |

Policy vs. Baseline A (net): ₹5,334,298.00
Policy vs. Baseline B (net): ₹2,541,209.00

Policy costs already subtracted into net: gateway ₹9,872.00, annoyance/friction ₹9,680.00
Baseline B costs already subtracted into net: gateway ₹12,006.00, annoyance/friction ₹21,965.00

## 1. False-Positive Retry Rate

Retries attempted on transactions where `gt_is_recoverable == False` — wasted/wrong retries.

All rows carried real gt_is_recoverable / gt_ideal_retry_action / gt_requires_customer_action labels — no fallback used.

Total transactions with `gt_is_recoverable == False`: 2,697

| | Wasted retries (n) | FP rate (of unrecoverable) | Amount at risk in wasted retries | Share of all retries fired |
|---|---|---|---|---|
| **Policy engine** | 1,708 | 63.33% | ₹2,855,018.00 | 34.60% |
| Baseline B (naive) | 2,697 | 100.00% | ₹4,715,573.00 | 44.93% |

FP rate = wasted retries / all `gt_is_recoverable == False` cases (standard false-positive-rate definition, FP / (FP+TN)). "Share of all retries fired" is a secondary view: what fraction of everything the policy attempted was wasted.

### Cost already deducted for these wasted retries

**Yes — the net recovered figure above already subtracts a per-retry cost for false-positive retries.** Every retry (gateway fee) and every *failed* retry (annoyance/friction cost) is charged into the net figure regardless of whether it was a wasted (false-positive) retry or a legitimate one; the table below isolates the slice of that already-applied deduction attributable specifically to the 1,708 wasted retries, so the effect is visible rather than buried in the aggregate.

| | Gateway fee cost | Annoyance/friction cost | Total cost from wasted retries |
|---|---|---|---|
| Policy engine | ₹3,416.00 | ₹3,865.00 | ₹7,281.00 |

Net recovered if these wasted retries had never been attempted (approximate — assumes their own revenue contribution is unchanged, since `gt_is_recoverable == False` implies it was already near-zero): **₹5,341,579.00** vs. the actual net of ₹5,334,298.00.

## 2. Guardrail Compliance (risk_block / card_expired)

**Status: PASS — zero retries recommended**

This verifies the Layer-1 invariant `retry_policy_engine.py` already enforces (`NO_RETRY_CATEGORIES` + `RISK_CODED_RAW_VALUES`) — it is not a new rule. Pass/fail is based on the check below that the invariant guarantees; the second number is real but is a different, expected-nonzero thing (classifier misclassification), not a guardrail failure.

- **Engine invariant (defines PASS/FAIL — should always be exactly 0):** 0 retries recommended when the engine itself saw a risk_block/card_expired prediction or risk-coded bank response code.
- **For context only — true-category leakage:** 13 retries were recommended on rows whose *true* category was risk_block/card_expired (1.36% of such rows), purely because the classifier misclassified them into a retry-eligible category before the engine ever saw them. This is a model-accuracy gap, not a guardrail bug — the engine can only gate on what it's told.

| Ground-truth category | n | Retries recommended (should be 0) |
|---|---|---|
| risk_block | 535 | 11 |
| card_expired | 422 | 2 |

Rows where `gt_is_recoverable`/`gt_ideal_retry_action` disagreed with a risk_block/card_expired `decline_reason_category` (data-quality flag, not a policy issue): 0

## 3. Latency-to-Recovery Distribution

Time from failed attempt to recovered success, bucketed, for recovered cases only.

### Policy engine
| Latency bucket | n | % of recovered |
|---|---|---|
| immediate (<5min) | 1,806 | 60.2% |
| <1hr | 1,185 | 39.5% |
| <1day | 9 | 0.3% |
| >1day | 0 | 0.0% |

### Baseline B (naive retry-once)
| Latency bucket | n | % of recovered |
|---|---|---|
| immediate (<5min) | 0 | 0.0% |
| <1hr | 0 | 0.0% |
| <1day | 1,610 | 100.0% |
| >1day | 0 | 0.0% |

(Raw stats — policy: n=3000, median=188s, p90=375s, p99=931s, max=47188s; baseline B: n=1610, median=3629s, p90=3663s, p99=3714s, max=3814s)

## Safety / Other Compliance Detail

- Hard-rule violations (Layer-1 invariant): 0
- Residual leakage (retries fired on truth-hard-blocked rows via misclassification): 13 (1.36%)
- Naive baseline hard-FP count/rate (always retries everything): 957 (100.0%)
- Over-block count/amount (hard rule fired on a non-hard-blocked truth category): 75 / ₹120,387.00 (expected recovery forgone: ₹75,844.23)
- "other" category policy gap (schema says no-automation, code doesn't hard-gate it): 328
- Flagged for human review: 640

`RISK_CODED_RAW_VALUES` has been customized from the placeholder set.

## Cost Assumptions & Sensitivity

Gateway fee assumption: ₹2.00 per retry attempt
Annoyance cost assumption: ₹5.00 per failed retry

| Scenario | Gateway fee | Annoyance cost | Net recovered (policy) | Net recovered (naive) | Policy advantage |
|---|---|---|---|---|---|
| baseline assumptions (1x fee / 1x annoyance) | ₹2.00 | ₹5.00 | ₹5,334,298.00 | ₹2,793,089.00 | ₹2,541,209.00 |
| gateway fee 2x | ₹4.00 | ₹5.00 | ₹5,324,426.00 | ₹2,781,083.00 | ₹2,543,343.00 |
| gateway fee 0.5x | ₹1.00 | ₹5.00 | ₹5,339,234.00 | ₹2,799,092.00 | ₹2,540,142.00 |
| annoyance cost 2x | ₹2.00 | ₹10.00 | ₹5,324,618.00 | ₹2,771,124.00 | ₹2,553,494.00 |
| annoyance cost 0.5x | ₹2.00 | ₹2.50 | ₹5,339,138.00 | ₹2,804,071.50 | ₹2,535,066.50 |
| both costs 2x (worst case) | ₹4.00 | ₹10.00 | ₹5,314,746.00 | ₹2,759,118.00 | ₹2,555,628.00 |
| both costs 0.5x (best case) | ₹1.00 | ₹2.50 | ₹5,344,074.00 | ₹2,810,074.50 | ₹2,533,999.50 |
