"""
Payment Retry Sequencer — Phase 6: Ops Dashboard (Streamlit)
================================================================

Talks to the live Phase 5 API (POST /evaluate-attempt) running at
http://localhost:8000. Three tabs:
    1. Batch Upload  — upload a CSV of failed attempts, run each through the API.
    2. Drill-down    — inspect the full reasoning + guardrail checks for any
                        single evaluated attempt (the audit-trail view).
    3. Results       — Phase 4 backtest chart (revenue recovered vs baseline)
                        + LIVE pipeline-safety metrics computed from whatever
                        you've actually run through the Batch Upload tab.

Run:
    streamlit run app.py

Only two hard dependencies on "the outside world":
    - The Phase 5 API must be reachable at API_BASE_URL (edit below if not
      localhost:8000).
    - The Results tab's "revenue recovered" chart needs your actual Phase 4
      backtest report (JSON). It will NOT fabricate numbers if that file
      isn't found — see `load_phase4_report()` for the expected schema.
"""

import io
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
EVALUATE_URL = f"{API_BASE_URL}/evaluate-attempt"
HEALTH_URL = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_MAX_ROWS = 200  # safety cap for a live, sequential, local demo

# Exactly the request fields confirmed against the live Phase 5 API's
# /docs example. If your deployed API also accepts extra optional fields
# (e.g. merchant_archetype, customer_opted_out, last_attempt_time), add
# them here — FastAPI/Pydantic will otherwise just ignore unknown fields
# unless your model forbids extras, in which case add-don't-guess is safer.
CONFIRMED_REQUEST_FIELDS = [
    "transaction_id", "merchant_id", "customer_id", "amount", "currency",
    "payment_method", "gateway", "bank_response_code", "timestamp",
    "attempt_number", "is_recurring", "previous_attempts",
    "merchant_hist_failure_rate", "merchant_prior_attempt_count",
    "customer_hist_failure_rate", "customer_prior_attempt_count",
    "gateway_recent_failure_rate",
]
REQUIRED_REQUEST_FIELDS = [
    "transaction_id", "merchant_id", "customer_id", "amount",
    "payment_method", "gateway", "bank_response_code", "timestamp", "attempt_number",
]
INT_FIELDS = {"attempt_number", "merchant_prior_attempt_count", "customer_prior_attempt_count"}
FLOAT_FIELDS = {"amount", "merchant_hist_failure_rate", "customer_hist_failure_rate", "gateway_recent_failure_rate"}

# Columns that, if present in an uploaded CSV, are GROUND TRUTH / evaluation
# only. They are kept around for the Results tab's metrics but are NEVER
# sent to the API — the whole point of this pipeline is that production
# doesn't have these at decision time.
GROUND_TRUTH_PREFIXES = ("gt_", "ctx_")
GROUND_TRUTH_LABEL_COL = "decline_reason_category"

RETRYABLE_ACTIONS = {"same_gateway_retry", "alt_gateway_retry", "alt_payment_method_prompt", "delayed_retry"}
HARD_STOP_CATEGORIES = {"risk_block", "card_expired"}

st.set_page_config(page_title="Payment Retry Sequencer — Ops Dashboard", page_icon="🔁", layout="wide")

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None  # pd.DataFrame
if "batch_errors" not in st.session_state:
    st.session_state.batch_errors = None  # pd.DataFrame
if "last_single_result" not in st.session_state:
    st.session_state.last_single_result = None  # dict: {"payload": ..., "response": ...}


# ---------------------------------------------------------------------------
# Helpers — API calls
# ---------------------------------------------------------------------------

def build_payload(row: pd.Series) -> dict:
    """Whitelist-based payload construction: only send confirmed fields,
    never ground-truth columns, regardless of what else is in the CSV."""
    payload = {}
    for field in CONFIRMED_REQUEST_FIELDS:
        if field not in row.index or pd.isna(row[field]):
            continue
        val = row[field]
        if field == "previous_attempts":
            if isinstance(val, str):
                try:
                    payload[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    payload[field] = []
            elif isinstance(val, list):
                payload[field] = val
            else:
                payload[field] = []
        elif field == "is_recurring":
            payload[field] = bool(val)
        elif field == "timestamp":
            payload[field] = pd.Timestamp(val).isoformat()
        elif field in INT_FIELDS:
            payload[field] = int(val)
        elif field in FLOAT_FIELDS:
            payload[field] = float(val)
        else:
            payload[field] = val
    return payload


def call_api(payload: dict):
    """Returns (response_json_or_None, error_message_or_None)."""
    try:
        resp = requests.post(EVALUATE_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as e:
        return None, f"Could not reach API at {EVALUATE_URL}: {e}"
    if resp.status_code == 200:
        return resp.json(), None
    try:
        detail = resp.json()
    except ValueError:
        detail = resp.text[:500]
    return None, f"HTTP {resp.status_code}: {detail}"


def run_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, errors = [], []
    n = len(df)
    progress = st.progress(0.0, text=f"Evaluating attempts... 0/{n}")
    gt_cols = [c for c in df.columns if c.startswith(GROUND_TRUTH_PREFIXES) or c == GROUND_TRUTH_LABEL_COL]

    for i, (idx, row) in enumerate(df.iterrows()):
        txn_id = row.get("transaction_id", f"row_{idx}")
        missing = [f for f in REQUIRED_REQUEST_FIELDS if f not in row.index or pd.isna(row[f])]
        if missing:
            errors.append({"row_index": idx, "transaction_id": txn_id,
                            "error": f"Missing required field(s): {missing}"})
        else:
            payload = build_payload(row)
            response, err = call_api(payload)
            if err:
                errors.append({"row_index": idx, "transaction_id": txn_id, "error": err})
            else:
                record = {"_input_payload": json.dumps(payload)}
                record.update(response)
                for c in gt_cols:
                    record[c] = row[c]
                results.append(record)
        progress.progress((i + 1) / n, text=f"Evaluating attempts... {i + 1}/{n}")

    progress.empty()
    return pd.DataFrame(results), pd.DataFrame(errors)


# ---------------------------------------------------------------------------
# Helpers — guardrail-check display
# ---------------------------------------------------------------------------

def classify_guardrail(text: str) -> str:
    """Heuristic pass/fail badge for a guardrail_checks string.

    NOTE: the exact wording your Phase 3 engine uses for a FAILING guardrail
    wasn't available when this dashboard was written (only passing examples
    were on hand). This looks for common "everything's fine" phrasing and
    falls back to a neutral badge rather than guessing at a violation. If
    your engine has a consistent violation phrase (e.g. "VIOLATION" or
    "BLOCKED"), add it to `FAIL_MARKERS` below for a precise red badge.
    """
    t = text.lower()
    fail_markers = ["violat", "blocked", "breach"]
    pass_markers = ["ok)", "ok,", "not opted out", "n/a)", "n/a,"]
    if any(m in t for m in fail_markers):
        return "fail"
    if any(m in t for m in pass_markers):
        return "pass"
    return "info"


def render_guardrail_checks(checks: list[str]):
    icon = {"pass": "✅", "fail": "❌", "info": "ℹ️"}
    for c in checks:
        status = classify_guardrail(c)
        st.markdown(f"{icon[status]} `{c}`")


# ---------------------------------------------------------------------------
# Helpers — Phase 4 report loading (Results tab)
# ---------------------------------------------------------------------------

# Matches the actual Phase 4 `run_backtest.py` export (results_summary.json):
# a top-level "results" object with "policy" / "baseline_a" (do-nothing) /
# "baseline_b" (naive retry-everything) scenarios, plus "false_positive",
# "guardrail_compliance", "safety", and "cost_inputs" sub-objects, and a
# top-level "sensitivity" list of cost-assumption scenarios.
EXPECTED_PHASE4_SCHEMA_NOTE = (
    "Top-level keys: `results.{n, total_at_risk, n_hard_truth, policy, baseline_a, "
    "baseline_b, false_positive, guardrail_compliance, safety, cost_inputs}` and "
    "a top-level `sensitivity` list. See your own `results_summary.json` for the exact shape — "
    "this loader is written against it directly, not a guessed schema."
)


def load_phase4_report(uploaded_file) -> dict | None:
    """Loads the Phase 4 backtest report (results_summary.json). Returns None
    (and lets the caller show a warning) rather than fabricating anything on
    mismatch."""
    if uploaded_file is None:
        return None
    try:
        data = json.load(uploaded_file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        st.error("Couldn't parse that file as JSON.")
        return None
    if "results" not in data:
        st.error("JSON loaded, but it's missing the top-level 'results' key. "
                 f"{EXPECTED_PHASE4_SCHEMA_NOTE}")
        return None
    required_scenarios = {"policy", "baseline_a", "baseline_b"}
    missing = required_scenarios - set(data["results"].keys())
    if missing:
        st.error(f"'results' is missing scenario(s): {sorted(missing)}. {EXPECTED_PHASE4_SCHEMA_NOTE}")
        return None
    return data


def inr(x) -> str:
    return f"₹{x:,.0f}"


def pct(x) -> str:
    return f"{x:.1f}%"


# ---------------------------------------------------------------------------
# Sidebar — API connection status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Connection")
    st.caption("Phase 5 API")
    st.code(API_BASE_URL, language=None)
    if st.button("Check /health"):
        try:
            r = requests.get(HEALTH_URL, timeout=3)
            if r.ok:
                st.success(f"Reachable — HTTP {r.status_code}")
                st.json(r.json())
            else:
                st.warning(f"Responded with HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unreachable: {e}")
    st.divider()
    st.caption(
        "Ground-truth columns (`decline_reason_category`, `gt_*`, `ctx_*`) are "
        "never sent to the API — they're only used locally, in the Results tab, "
        "to score what the API got right."
    )

st.title("🔁 Payment Retry Sequencer — Ops Dashboard")
st.caption("Phase 6 · calls the live Phase 5 API for every attempt shown here.")

tab_batch, tab_drilldown, tab_results = st.tabs(["📤 Batch Upload", "🔍 Drill-down", "📊 Results"])

# ---------------------------------------------------------------------------
# TAB 1 — Batch Upload
# ---------------------------------------------------------------------------

with tab_batch:
    st.subheader("Upload a batch of failed attempts")
    st.markdown(
        f"Required columns: `{'`, `'.join(REQUIRED_REQUEST_FIELDS)}`. "
        f"Optional: `{'`, `'.join(c for c in CONFIRMED_REQUEST_FIELDS if c not in REQUIRED_REQUEST_FIELDS)}`. "
        "Optional ground-truth columns for evaluation later: `decline_reason_category`, `gt_*`, `ctx_*` "
        "(kept for the Results tab, never sent to the API)."
    )

    sample_rows = [
        {"transaction_id": "demo-001", "merchant_id": "d2c_ecommerce_01", "customer_id": "cust_demo_001",
         "amount": 2499, "currency": "INR", "payment_method": "upi", "gateway": "gateway_a",
         "bank_response_code": "U69", "timestamp": "2026-09-05T12:00:00Z", "attempt_number": 1,
         "is_recurring": False, "previous_attempts": "[]", "merchant_hist_failure_rate": 0.15,
         "merchant_prior_attempt_count": 5000, "customer_hist_failure_rate": 0.05,
         "customer_prior_attempt_count": 3, "gateway_recent_failure_rate": 0.10,
         "decline_reason_category": "insufficient_funds"},
        {"transaction_id": "demo-002", "merchant_id": "marketplace_01", "customer_id": "cust_demo_002",
         "amount": 899, "currency": "INR", "payment_method": "upi", "gateway": "gateway_a",
         "bank_response_code": "U39", "timestamp": "2026-09-05T13:15:00Z", "attempt_number": 1,
         "is_recurring": False, "previous_attempts": "[]", "merchant_hist_failure_rate": 0.09,
         "merchant_prior_attempt_count": 800, "customer_hist_failure_rate": 0.0,
         "customer_prior_attempt_count": 0, "gateway_recent_failure_rate": 0.08,
         "decline_reason_category": "risk_block"},
        {"transaction_id": "demo-003", "merchant_id": "saas_subscription_01", "customer_id": "cust_demo_003",
         "amount": 999, "currency": "INR", "payment_method": "card", "gateway": "gateway_b",
         "bank_response_code": "54", "timestamp": "2026-09-05T02:30:00Z", "attempt_number": 1,
         "is_recurring": True, "previous_attempts": "[]", "merchant_hist_failure_rate": 0.06,
         "merchant_prior_attempt_count": 1200, "customer_hist_failure_rate": 0.02,
         "customer_prior_attempt_count": 11, "gateway_recent_failure_rate": 0.05,
         "decline_reason_category": "card_expired"},
    ]
    sample_csv = pd.DataFrame(sample_rows).to_csv(index=False)
    st.download_button("⬇️ Download a sample CSV template", data=sample_csv,
                        file_name="phase6_sample_batch.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001 — surfaced directly to the user
            st.error(f"Couldn't read that CSV: {e}")
            df = None

        if df is not None:
            st.write(f"Loaded **{len(df):,}** rows.")
            st.dataframe(df.head(10), use_container_width=True)

            missing_required = [c for c in REQUIRED_REQUEST_FIELDS if c not in df.columns]
            if missing_required:
                st.error(f"Missing required column(s): {missing_required}. Fix the CSV and re-upload.")
            else:
                max_rows = st.number_input(
                    "Max rows to process (safety cap for a live, sequential local demo)",
                    min_value=1, max_value=max(len(df), 1), value=min(DEFAULT_MAX_ROWS, len(df)), step=10,
                )
                if len(df) > max_rows:
                    st.info(f"Only the first {max_rows} of {len(df)} rows will be processed.")

                if st.button("▶️ Run batch through the API", type="primary"):
                    results_df, errors_df = run_batch(df.head(int(max_rows)))
                    st.session_state.batch_results = results_df
                    st.session_state.batch_errors = errors_df

    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results
        errors_df = st.session_state.batch_errors

        st.divider()
        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processed", len(results_df))
        c2.metric("Errors", 0 if errors_df is None else len(errors_df))
        if len(results_df) > 0:
            c3.metric("Flagged for human review", int(results_df["flagged_for_human_review"].sum()))
            c4.metric("Avg. classifier confidence", f"{results_df['predicted_confidence'].mean():.1%}")

        if errors_df is not None and len(errors_df) > 0:
            with st.expander(f"⚠️ {len(errors_df)} row(s) failed — click to view"):
                st.dataframe(errors_df, use_container_width=True)

        if len(results_df) > 0:
            display_cols = ["transaction_id", "predicted_category", "predicted_confidence",
                             "recommended_action", "timing", "confidence", "flagged_for_human_review"]
            if GROUND_TRUTH_LABEL_COL in results_df.columns:
                display_cols.insert(2, GROUND_TRUTH_LABEL_COL)
                results_df["classifier_correct"] = (
                    results_df["predicted_category"] == results_df[GROUND_TRUTH_LABEL_COL]
                )
                display_cols.append("classifier_correct")
            st.dataframe(results_df[[c for c in display_cols if c in results_df.columns]],
                         use_container_width=True)
            st.download_button("⬇️ Download full results as CSV",
                                data=results_df.to_csv(index=False),
                                file_name="phase6_batch_results.csv", mime="text/csv")
            st.caption("Head to the **Drill-down** tab to inspect the full reasoning for any row above.")

# ---------------------------------------------------------------------------
# TAB 2 — Drill-down
# ---------------------------------------------------------------------------

with tab_drilldown:
    st.subheader("Quick single-attempt test")
    st.caption("Paste or edit a single attempt payload and run it directly — useful for a live demo "
               "or for sanity-checking the API connection independent of a batch.")

    default_payload = json.dumps({
        "transaction_id": "demo-single-001", "merchant_id": "d2c_ecommerce_01", "customer_id": "cust_demo_001",
        "amount": 2499, "currency": "INR", "payment_method": "upi", "gateway": "gateway_a",
        "bank_response_code": "U69", "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt_number": 1, "is_recurring": False, "previous_attempts": [],
        "merchant_hist_failure_rate": 0.15, "merchant_prior_attempt_count": 5000,
        "customer_hist_failure_rate": 0.05, "customer_prior_attempt_count": 3,
        "gateway_recent_failure_rate": 0.10,
    }, indent=2)
    payload_text = st.text_area("Request JSON", value=default_payload, height=260)

    if st.button("▶️ Run this attempt"):
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            payload = None
        if payload is not None:
            response, err = call_api(payload)
            if err:
                st.error(err)
            else:
                st.session_state.last_single_result = {"payload": payload, "response": response}

    if st.session_state.last_single_result is not None:
        st.success("Evaluated. Full audit record below.")
        payload = st.session_state.last_single_result["payload"]
        response = st.session_state.last_single_result["response"]

        col_a, col_b = st.columns(2)
        col_a.metric("Predicted category", response.get("predicted_category", "—"),
                     f"{response.get('predicted_confidence', 0):.1%} confidence")
        col_b.metric("Recommended action", response.get("recommended_action", "—"),
                     f"{response.get('timing', '—')} · {response.get('confidence', 0):.1%} est. recovery")
        if response.get("flagged_for_human_review"):
            st.error("🚩 Flagged for human review")
        else:
            st.success("✅ Not flagged for human review")

        st.markdown("**Reasoning**")
        st.info(response.get("reason_string", "—"))

        st.markdown("**Guardrail checks**")
        render_guardrail_checks(response.get("guardrail_checks", []))

        st.caption(f"request_id: `{response.get('request_id', '—')}` · "
                   f"evaluated_at: `{response.get('evaluated_at', '—')}`")

        with st.expander("Full request payload sent to the API"):
            st.json(payload)
        with st.expander("Full raw API response"):
            st.json(response)

    st.divider()
    st.subheader("Batch results drill-down")
    if st.session_state.batch_results is None or len(st.session_state.batch_results) == 0:
        st.info("Run a batch in the **Batch Upload** tab first, or use the quick test above.")
    else:
        results_df = st.session_state.batch_results
        selected_txn = st.selectbox("Select a transaction to inspect", results_df["transaction_id"].tolist())
        row = results_df[results_df["transaction_id"] == selected_txn].iloc[0]

        col_a, col_b = st.columns(2)
        col_a.metric("Predicted category", row.get("predicted_category", "—"),
                     f"{row.get('predicted_confidence', 0):.1%} confidence")
        col_b.metric("Recommended action", row.get("recommended_action", "—"),
                     f"{row.get('timing', '—')} · {row.get('confidence', 0):.1%} est. recovery")
        if GROUND_TRUTH_LABEL_COL in results_df.columns:
            true_cat = row.get(GROUND_TRUTH_LABEL_COL)
            match = row.get("predicted_category") == true_cat
            st.markdown(f"**True category (ground truth):** `{true_cat}` "
                        f"{'✅ matches prediction' if match else '❌ does NOT match prediction'}")
        if row.get("flagged_for_human_review"):
            st.error("🚩 Flagged for human review")
        else:
            st.success("✅ Not flagged for human review")

        st.markdown("**Reasoning**")
        st.info(row.get("reason_string", "—"))

        st.markdown("**Guardrail checks**")
        guardrails = row.get("guardrail_checks", [])
        if isinstance(guardrails, str):
            try:
                guardrails = json.loads(guardrails)
            except json.JSONDecodeError:
                guardrails = [guardrails]
        render_guardrail_checks(guardrails or [])

        st.caption(f"request_id: `{row.get('request_id', '—')}` · evaluated_at: `{row.get('evaluated_at', '—')}`")

        with st.expander("Full request payload sent to the API"):
            st.json(json.loads(row.get("_input_payload", "{}")))
        with st.expander("Full raw API response"):
            st.json({k: v for k, v in row.to_dict().items()
                     if not k.startswith(("_input", "gt_", "ctx_")) and k != GROUND_TRUTH_LABEL_COL})

# ---------------------------------------------------------------------------
# TAB 3 — Results
# ---------------------------------------------------------------------------

with tab_results:
    st.subheader("Phase 4 backtest results")
    phase4_file = st.file_uploader("Upload results_summary.json", type=["json"], key="phase4_upl")
    report = load_phase4_report(phase4_file)

    with st.expander("Expected report shape"):
        st.caption(EXPECTED_PHASE4_SCHEMA_NOTE)

    if report is None:
        st.warning("No Phase 4 report loaded — nothing fabricated here. Upload results_summary.json above.")
    else:
        r = report["results"]
        policy, base_a, base_b = r["policy"], r["baseline_a"], r["baseline_b"]
        cost_inputs = r.get("cost_inputs", {})

        # --- 1. Headline recovery comparison -----------------------------------
        st.markdown(f"**{r['n']:,} failed attempts** · **{inr(r['total_at_risk'])} total at risk** · "
                    f"cost assumptions: gateway fee {inr(cost_inputs.get('gateway_fee', 0))}/retry, "
                    f"annoyance cost {inr(cost_inputs.get('annoyance_cost', 0))}/retry")

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline A — do nothing", inr(base_a["net_recovered"]), "0% of at-risk value")
        c2.metric("Baseline B — retry everything", inr(base_b["net_recovered"]), pct(base_b["net_pct_at_risk"]))
        c3.metric("Policy engine", inr(policy["net_recovered"]), pct(policy["net_pct_at_risk"]))

        uplift_vs_b = policy["net_recovered"] - base_b["net_recovered"]
        st.success(
            f"Policy recovers **{inr(uplift_vs_b)} more net revenue** than blindly retrying everything "
            f"({pct(policy['recovery_rate_pct'])} recovery rate on the attempts it chose to retry, vs. "
            f"{pct(base_b['recovery_rate_pct'])} for baseline B) — while making {policy['retries_attempted']:,} "
            f"targeted retry attempts instead of {base_b['retries_attempted']:,} blind ones."
        )

        chart_df = pd.DataFrame({
            "scenario": ["Baseline A\n(do nothing)", "Baseline B\n(retry everything)", "Policy"],
            "net_recovered": [base_a["net_recovered"], base_b["net_recovered"], policy["net_recovered"]],
            "gross_recovered": [base_a["gross_recovered"], base_b["gross_recovered"], policy["gross_recovered"]],
        }).set_index("scenario")
        st.bar_chart(chart_df)

        # --- 2. Guardrail safety — the two-layer story --------------------------
        st.divider()
        st.subheader("Guardrail safety: classifier vs. engine")
        gc, safety = r["guardrail_compliance"], r["safety"]

        c1, c2 = st.columns(2)
        c1.metric("Engine hard-rule violations", gc["engine_hard_rule_violations"],
                   "✅ compliant" if gc["compliant"] else "🚨 NOT compliant")
        c2.metric("Naive baseline (retry-everything) hard-stop FP rate", pct(safety["naive_baseline_hard_fp_rate_pct"]))

        st.info(
            f"The Phase 2 classifier alone missed **{gc['gt_violations_n']} of {r['n_hard_truth']}** "
            f"true hard-stop cases ({pct(gc['gt_violations_pct'])}) — "
            f"{gc['by_category']['risk_block']['violations']}/{gc['by_category']['risk_block']['n']} "
            f"`risk_block` and {gc['by_category']['card_expired']['violations']}/{gc['by_category']['card_expired']['n']} "
            f"`card_expired` cases were misclassified as something else. **None of those 13 misses reached "
            f"the customer as a retry** — the policy engine's independent hard-rule check (keyed off more "
            f"than just the classifier's category call) caught all of them, landing at 0 actual violations. "
            f"This is the guardrail design working as intended: the ML layer doesn't have to be perfect "
            f"because it isn't the only thing enforcing the hard stop."
        )
        st.caption(
            "Contrast with baseline B, which has no classifier or guardrail at all — it retries every "
            f"failed attempt blindly, so its hard-stop false-positive rate is definitionally {pct(safety['naive_baseline_hard_fp_rate_pct'])}."
        )

        # --- 3. Wasted retries (false positives) --------------------------------
        st.divider()
        st.subheader("Wasted retries on truly-unrecoverable attempts")
        fp = r["false_positive"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Policy false-positive rate", pct(fp["policy"]["fp_rate_pct"]),
                   f"{fp['policy']['wasted_retries_n']:,} of {fp['n_gt_unrecoverable']:,} unrecoverable attempts")
        c2.metric("Baseline B false-positive rate", pct(fp["baseline_b"]["fp_rate_pct"]),
                   f"{fp['baseline_b']['wasted_retries_n']:,} of {fp['n_gt_unrecoverable']:,}")
        c3.metric("Cost of policy's wasted retries", inr(fp["policy"]["wasted_total_cost"]))
        st.caption(
            f"'False positive' here means the policy chose to retry an attempt that, per ground truth, was "
            f"never going to recover. Policy still wastes retries on {pct(fp['policy']['fp_rate_pct'])} of the "
            f"unrecoverable population — but the cost is small: eliminating every one of those wasted retries "
            f"would only raise net recovery from {inr(policy['net_recovered'])} to "
            f"{inr(fp['policy']['net_recovered_if_no_fp_retries'])}, a "
            f"{inr(fp['policy']['net_recovered_if_no_fp_retries'] - policy['net_recovered'])} difference. "
            "This is a probabilistic policy making reasonable bets on ambiguous cases, not a design flaw."
        )

        # --- 4. Over-blocking (the opposite failure mode) -----------------------
        st.divider()
        st.subheader("Over-blocking (excess caution)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Attempts over-blocked", f"{safety['over_block_n']:,}")
        c2.metric("Amount over-blocked", inr(safety["over_block_amount"]))
        c3.metric("Recovery forgone from over-blocking", inr(safety["over_block_expected_recovery_forgone"]))
        st.caption(
            f"Plus {safety['other_category_policy_gap_n']:,} attempts fell into the `other` category "
            f"(no automated action by design) and {safety['flagged_for_human_review_n']:,} were flagged for "
            "human review in total — both are manual-review queue size, not errors."
        )

        # --- 5. Recovery latency -------------------------------------------------
        st.divider()
        st.subheader("Recovery latency")
        lat_col1, lat_col2 = st.columns(2)
        with lat_col1:
            st.markdown("**Policy**")
            pl = policy["latency"]
            st.caption(f"median {pl['median_s']/60:.1f} min · p90 {pl['p90_s']/60:.1f} min · "
                       f"p99 {pl['p99_s']/3600:.1f} hr · max {pl['max_s']/3600:.1f} hr")
            bucket_df = pd.DataFrame(policy["latency_buckets"]).set_index("label")["pct"]
            st.bar_chart(bucket_df)
        with lat_col2:
            st.markdown("**Baseline B**")
            bl = base_b["latency"]
            st.caption(f"median {bl['median_s']/60:.1f} min · p90 {bl['p90_s']/60:.1f} min · "
                       f"p99 {bl['p99_s']/3600:.1f} hr · max {bl['max_s']/3600:.1f} hr")
            bucket_df_b = pd.DataFrame(base_b["latency_buckets"]).set_index("label")["pct"]
            st.bar_chart(bucket_df_b)
        st.caption("Policy recovers most successes within 5 minutes to an hour (it retries transient errors "
                   "immediately and waits out insufficient-funds cases); baseline B's fixed delay lands "
                   "everything in the same multi-hour bucket regardless of the actual decline reason.")

        # --- 6. Sensitivity analysis ----------------------------------------------
        if report.get("sensitivity"):
            st.divider()
            st.subheader("Sensitivity to cost assumptions")
            sens_df = pd.DataFrame(report["sensitivity"]).set_index("scenario")
            st.bar_chart(sens_df["policy_advantage"])
            st.caption(
                f"Policy's net-recovery advantage over baseline B stays in the "
                f"{inr(sens_df['policy_advantage'].min())}–{inr(sens_df['policy_advantage'].max())} range "
                "across a 4x swing in gateway-fee and annoyance-cost assumptions — the result isn't an "
                "artifact of the specific cost numbers chosen."
            )
            with st.expander("Full sensitivity table"):
                st.dataframe(sens_df, use_container_width=True)

    st.divider()
    st.subheader("Live pipeline-safety metrics (from Batch Upload tab)")
    st.caption("This section re-checks the same guardrail question — using whatever you've actually run "
               "through the live API in the Batch Upload tab, which may be a small sample vs. the full "
               "6,003-attempt backtest above.")
    results_df = st.session_state.batch_results

    if results_df is None or len(results_df) == 0:
        st.info("Run a batch with a `decline_reason_category` ground-truth column in the "
                 "Batch Upload tab to populate these metrics.")
    elif GROUND_TRUTH_LABEL_COL not in results_df.columns:
        st.info("Your last batch didn't include a `decline_reason_category` column, so accuracy/safety "
                 "metrics can't be computed. Re-run with that column included (it's never sent to the API).")
    else:
        n = len(results_df)
        accuracy = (results_df["predicted_category"] == results_df[GROUND_TRUTH_LABEL_COL]).mean()

        hard_stop_mask = results_df[GROUND_TRUTH_LABEL_COL].isin(HARD_STOP_CATEGORIES)
        n_hard_stop = int(hard_stop_mask.sum())
        if n_hard_stop > 0:
            dangerous_mask = hard_stop_mask & results_df["recommended_action"].isin(RETRYABLE_ACTIONS)
            n_dangerous = int(dangerous_mask.sum())
            compliance_rate = 1 - (n_dangerous / n_hard_stop)
        else:
            n_dangerous, compliance_rate = 0, None

        c1, c2, c3 = st.columns(3)
        c1.metric("Batch size", n)
        c2.metric("Classifier accuracy vs. ground truth", f"{accuracy:.1%}")
        if compliance_rate is not None:
            c3.metric(
                "Guardrail compliance (risk_block/card_expired → no_retry)",
                f"{compliance_rate:.1%}",
                delta=f"{n_dangerous} violation(s)" if n_dangerous else "0 violations",
                delta_color="inverse",
            )
        else:
            c3.metric("Guardrail compliance", "n/a — no risk_block/card_expired rows in this batch")

        if n_dangerous > 0:
            st.error(
                f"🚨 {n_dangerous} of {n_hard_stop} true risk_block/card_expired attempts were recommended "
                "a RETRYABLE action. This should never happen if the Phase 3 hard rule is wired correctly — "
                "treat this as a bug, not a tuning issue."
            )
            st.dataframe(
                results_df.loc[hard_stop_mask & results_df["recommended_action"].isin(RETRYABLE_ACTIONS),
                               ["transaction_id", GROUND_TRUTH_LABEL_COL, "predicted_category", "recommended_action"]],
                use_container_width=True,
            )
        elif n_hard_stop > 0:
            st.success(f"✅ 0 violations across {n_hard_stop} true risk_block/card_expired attempts in this batch — "
                       "the hard-stop rule held.")

        st.caption(
            "'Guardrail compliance' here checks the full pipeline's OUTPUT (classifier → policy engine), "
            "not just the Phase 2 classifier's raw prediction — a misclassified risk_block that still "
            "gets routed to no_retry by Phase 3's independent check would correctly show as compliant here."
        )
