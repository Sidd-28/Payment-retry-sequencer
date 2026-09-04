# Compliance Notes — Phase 7: Audit Trail & Guardrail Enforcement

This is a defensive engineering exercise (a synthetic "Payment Retry
Sequencer"), not a system connected to a live payment processor or real
customer data. These notes describe the data boundaries and design
rationale for Phase 7 specifically, on top of the Phase 3 policy engine
and Phase 5 API it wraps.

## 1. What this system does NOT use

- **No scraped or third-party-enriched data.** Every input to the policy
  engine and the audit log is a value already produced inside this
  pipeline (Phase 1's synthetic transaction generator, Phase 2's
  classifier). Nothing here calls out to social media, data brokers,
  credit bureaus, or any external enrichment source.
- **No payment credentials.** No card number (PAN), CVV, expiry date, or
  bank account/routing number is read, stored, or logged at any layer.
  `ClassifierOutput` and `TransactionContext` (Phase 3) carry only a
  decline category, a confidence score, a coded bank response, and
  transaction metadata (attempt count, timestamps, gateway health,
  amount, payment *method* as a category like "card" — never the
  instrument itself).
- **No PII beyond a synthetic, already-hashed/opaque identifier.** The
  only per-customer/per-transaction handle that flows through the
  system is `transaction_id`, which upstream phases generate as a
  synthetic or hashed token, not a name, email, phone number, or
  government ID. The audit log persists exactly what it's given — it
  does not add, infer, or look up any additional identifying
  information about the customer.
- **No free-text customer input.** All fields are structured
  (enums, floats, booleans, ISO timestamps). There is no chat
  transcript, support-ticket text, or other freeform field that could
  smuggle in PII or unrelated sensitive content.

## 2. Why the design is defense-in-depth (guardrails outside the model)

The hard-stop rules (`risk_block`, `card_expired`, a risk-coded raw bank
response, a customer's opt-out, and the attempt cap) are enforced by
**plain, reviewable code that sits outside the model's weights**, at
three independent points, not by training the model to "know better":

1. **Phase 3, Layer 1 (`apply_hard_rules`)** — runs before the ML/
   heuristic scoring layer even sees the case. A `risk_block` or
   `card_expired` category never reaches the recovery-probability
   estimator at all.
2. **Phase 3, Layer 3 (`apply_guardrails`)** — runs after the ML layer's
   pick, and can only make the outcome more conservative (shrink an
   aggressive action, push it later, or collapse it to `no_retry`).
3. **Phase 7 (`guardrail_enforcement.enforce_guardrails`)** — an
   additional, independent layer around *any* policy function. Critically,
   this layer does **not** trust the `PolicyDecision` object it receives.
   It re-derives the hard-stop verdict directly from the raw
   `ClassifierOutput`/`TransactionContext` inputs
   (`independent_hard_stop_reason`), the same way Layer 1 does, but as a
   *second, separately-coded implementation*. If the wrapped function is
   buggy, has been swapped for a different model, or is mocked out
   entirely in a test, this layer still fires.

**Why this matters more than "training it into the model":** a model's
behavior is a statistical fit to training data. It can drift on
retraining, be fooled by inputs designed to steer it toward a permissive
label, or simply be wrong in ways that are hard to predict in advance —
Phase 2's own evaluation found a non-zero dangerous-misclassification
rate on exactly the categories this guardrail protects. A rule that
lives in ordinary source code, under normal code review and version
control, with unit tests that assert it against adversarial and
malformed inputs (see `test_guardrail_enforcement.py`), gives a much
stronger and more auditable guarantee than hoping a model has
internalized an equivalent constraint. Putting the same check at three
independent layers means a single bug or a single compromised layer is
not sufficient to let a hard-stop case slip through — every layer would
have to fail the same way at once.

The corollary, tested explicitly: the guardrail layer is **downgrade-only**.
It can turn a risky recommendation into a safer one, but it can never
turn a safe recommendation into a riskier one — asserted as an explicit
invariant in `guardrail_enforcement.py`, not just an emergent property of
the logic.

## 3. Fail-safe behavior

Two situations that are easy to get wrong are handled explicitly:

- **Malformed input** (wrong type, out-of-range confidence, negative
  attempt count, etc.) is rejected *before* it reaches the policy
  function, and fails to `no_retry` with `flagged_for_human_review=True`
  — never a crash, and never a silent pass-through with default/guessed
  values.
- **An exception raised by the wrapped policy function itself** (e.g. a
  model-serving error) is caught and also fails to `no_retry`, flagged
  for review, with the exception recorded in the audit trail
  (`event_type: "policy_fn_exception"`).

In both cases the system fails *closed* (stops attempting the payment
and asks for a human) rather than *open* (attempts it anyway).

## 4. Audit trail immutability — what's guaranteed here vs. in production

`audit_log.AuditLogger` gives two guarantees achievable in pure Python
inside this sandbox:

- **Append-only interface.** The class exposes exactly one write method,
  `record()`. There is no `update`/`delete`/`truncate` method anywhere
  on the object — the application code has no way to call something
  that doesn't exist. Every physical write uses `O_APPEND`, a POSIX
  guarantee that writes land at end-of-file atomically.
- **Tamper-evidence via hash chaining.** Each line embeds a SHA-256 hash
  of its own content plus the previous line's hash. `verify_chain()`
  recomputes this from scratch and will flag the exact record where an
  edit, deletion, or reordering occurred — including edits made by
  hand-modifying the file outside this class entirely (see
  `test_audit_log_detects_tampering`).

This is tamper-**evident**, not tamper-**proof**: someone with direct
filesystem access and enough effort could still rewrite the whole file
and regenerate a self-consistent (but fabricated) chain from scratch.
Closing that gap requires infrastructure controls outside of any Python
process — for example `chattr +a` (Linux append-only file attribute),
writing to a versioned bucket with object-lock/WORM enabled (e.g. S3
Object Lock in compliance mode), or synchronously shipping each record
to a dedicated append-only logging service with its own access
controls. Phase 7 is the application-level half of that story; the
infra-level half is a deployment decision outside this codebase's scope,
and is called out here rather than silently assumed.

## 5. Scope note

This module intentionally does not change anything in Phase 3
(`retry_policy_engine.py`). It wraps it. Phase 3's own hard-rule and
guardrail layers keep running unmodified whether or not Phase 7's
wrapper is in the call path — which is exactly what
`test_direct_call_*_bypassing_wrapper_still_blocks` verifies.

## Known limitation: residual risk_block/card_expired leakage

1.36% residual leakage on risk_block/card_expired persists even after adding real raw-code cross-checks, because ~8% of true risk_block cases are assigned a raw code from an unrelated category by the data generator's noise model — no signal survives for the engine to catch these. This is a structural limit of probabilistic classification, not a fixable guardrail bug.