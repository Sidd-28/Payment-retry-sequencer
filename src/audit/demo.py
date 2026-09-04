"""
Phase 7 demo: wraps Phase 3's `choose_action` with the guardrail +
audit-log wrapper and runs it over a few scenarios, including one where
a stand-in "adversarial" model always recommends the riskiest action.

Run:  python3 demo.py
"""

import json

from retry_policy_engine import ClassifierOutput, TransactionContext, choose_action, RetryAction, PolicyDecision
from audit_log import AuditLogger
from guardrail_enforcement import enforce_guardrails


def adversarial_model(clf, ctx):
    """Always recommends the most aggressive action, ignoring everything."""
    return PolicyDecision(
        recommended_action=RetryAction.SAME_GATEWAY_RETRY.value,
        timing="immediate",
        confidence=0.99,
        reason_string="adversarial stand-in model: always retry",
        guardrail_checks=[],
        flagged_for_human_review=False,
    )


def main():
    audit_logger = AuditLogger("demo_audit_log.jsonl")
    safe_choose_action = enforce_guardrails(audit_logger)(choose_action)
    safe_adversarial = enforce_guardrails(audit_logger)(adversarial_model)

    scenarios = [
        ("normal, retry-eligible case", safe_choose_action,
         ClassifierOutput("network_timeout", 0.88, "U71"),
         TransactionContext("txn_demo_1", attempt_count=1)),
        ("real model would already block this (risk_block)", safe_choose_action,
         ClassifierOutput("risk_block", 0.95, "U39"),
         TransactionContext("txn_demo_2", attempt_count=1)),
        ("adversarial model tries to retry a risk_block anyway", safe_adversarial,
         ClassifierOutput("risk_block", 0.99, "U39"),
         TransactionContext("txn_demo_3", attempt_count=1)),
        ("adversarial model ignores customer opt-out", safe_adversarial,
         ClassifierOutput("network_timeout", 0.99, "U71"),
         TransactionContext("txn_demo_4", attempt_count=1, customer_opted_out=True)),
        ("malformed input (bad confidence type)", safe_choose_action,
         ClassifierOutput("network_timeout", "very confident", "U71"),
         TransactionContext("txn_demo_5", attempt_count=1)),
    ]

    for label, fn, clf, ctx in scenarios:
        decision = fn(clf, ctx)
        print(f"\n=== {label} ===")
        print(json.dumps(decision.to_dict(), indent=2))

    ok, bad_seq = audit_logger.verify_chain()
    print(f"\nAudit log chain intact: {ok} (first broken seq if any: {bad_seq})")
    print(f"Total audit records written: {sum(1 for _ in audit_logger.iter_records())}")


if __name__ == "__main__":
    main()
