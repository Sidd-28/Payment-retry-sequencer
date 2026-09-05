"""
Payment Retry Sequencer — Phase 5: FastAPI Service
=====================================================

Wraps the Phase 2 classifier (`score_with_phase2_model.py`'s model-loading
logic) and the Phase 3 policy engine (`retry_policy_engine.py`) behind a
small HTTP API:

    POST /evaluate-attempt   failed-payment-attempt JSON -> policy decision
    GET  /health             basic liveness / model-loaded check

Every request/response pair is appended as one JSON line to a local audit
log (feeds Phase 7). The model is loaded once at startup, not per request.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then see the interactive docs at http://localhost:8000/docs

--------------------------------------------------------------------------
REQUIRED FILES ALONGSIDE THIS ONE (same directory, or importable on PYTHONPATH):
    retry_policy_engine.py       (Phase 3 — unmodified)
    score_with_phase2_model.py   (Phase 4 — unmodified; supplies load_model,
                                   find_model_file, LoadedModel,
                                   CATEGORICAL_COLUMNS, CLASS_ORDER_FALLBACK)
    feature_engineering.py       (new, Phase 5 — see its docstring: some
                                   features are TODO / not yet portable
                                   from build_phase2_features.py)

CONFIG (environment variables, all optional):
    PHASE2_MODEL_DIR        directory to search for a model file
                             (default "docs/phase2_plots", same default as
                             score_with_phase2_model.py)
    PHASE2_MODEL_PATH       explicit model file path, skips the dir search
    PHASE2_CLASS_ORDER      comma-separated class order override, only
                             needed for a raw Booster with no --class-order
                             equivalent already confirmed
    AUDIT_LOG_PATH          where to append JSONL audit records
                             (default "./audit_log.jsonl")
    AMOUNT_BUCKET_ARTIFACTS_PATH
                             path to the frozen amount_bucket bin edges +
                             labels JSON produced by
                             `build_phase2_features.py --save-artifacts`.
                             Required if the loaded model needs
                             amount_bucket (see feature_engineering.py).
    ALLOW_UNKNOWN_FEATURES  "1" to start anyway when the model needs a
                             feature feature_engineering.py can't build yet
                             (logs a loud warning instead of refusing to
                             start — for local dev only, never in prod)
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from feature_engineering import (
    build_feature_row,
    configure_amount_bucket_artifacts,
    unresolved_features,
)
from retry_policy_engine import ClassifierOutput, TransactionContext, evaluate_retry_policy
from score_with_phase2_model import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER_FALLBACK,
    LoadedModel,
    find_model_file,
    load_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.environ.get("PHASE2_MODEL_DIR", "docs/phase2_plots"))
MODEL_PATH = os.environ.get("PHASE2_MODEL_PATH")
CLASS_ORDER_ENV = os.environ.get("PHASE2_CLASS_ORDER")
AUDIT_LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", "./audit_log.jsonl"))
ALLOW_UNKNOWN_FEATURES = os.environ.get("ALLOW_UNKNOWN_FEATURES", "0") == "1"
AMOUNT_BUCKET_ARTIFACTS_PATH = os.environ.get("AMOUNT_BUCKET_ARTIFACTS_PATH")

_audit_lock = threading.Lock()


# ---------------------------------------------------------------------------
# App state — populated once at startup, read (never mutated) per request
# ---------------------------------------------------------------------------

class ModelState:
    model: Optional[LoadedModel] = None
    class_order: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    model_path: Optional[str] = None
    loaded_at: Optional[datetime] = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup: load model once, fail fast on anything unsafe ----
    model_path = find_model_file(MODEL_DIR, MODEL_PATH)
    model = load_model(model_path)

    # CLASS_ORDER_FALLBACK (imported from score_with_phase2_model.py) is
    # ["insufficient_funds", "bank_server_error", "invalid_otp", "risk_block",
    #  "network_timeout", "card_expired", "limit_exceeded", "other"] —
    # confirmed correct against train_classifier.py's DECLINE_REASONS, per
    # that file's own docstring. Used explicitly here rather than branching
    # on model.classes_from_model at runtime, since this model is a raw
    # Booster with no .classes_ anyway. Set PHASE2_CLASS_ORDER only if you
    # swap in a genuinely different trained model.
    class_order = CLASS_ORDER_ENV.split(",") if CLASS_ORDER_ENV else CLASS_ORDER_FALLBACK

    if model.classes_from_model and model.classes_from_model != class_order:
        # Non-blocking: the model itself reports a different class order
        # than the one we're about to use. Only possible if a sklearn-API
        # model got swapped in later — flag it loudly rather than silently
        # trusting either side.
        print(
            f"[startup] WARNING: model.classes_ = {model.classes_from_model} "
            f"does not match the class_order this service is using "
            f"({class_order}). Using {class_order} anyway — set "
            f"PHASE2_CLASS_ORDER to override if this model file is genuinely "
            f"different from the one CLASS_ORDER_FALLBACK was verified against.",
            file=sys.stderr,
        )

    missing = unresolved_features(model.feature_names)
    if missing:
        msg = (
            f"Model at {model_path} requires feature(s) with no registered "
            f"builder: {missing}. Add these to "
            f"feature_engineering.KNOWN_FEATURE_BUILDERS before serving "
            f"traffic — see that file's docstring."
        )
        if ALLOW_UNKNOWN_FEATURES:
            print(f"[startup] WARNING (ALLOW_UNKNOWN_FEATURES=1): {msg}", file=sys.stderr)
        else:
            raise SystemExit(f"[startup] REFUSING TO START: {msg}")

    if "amount_bucket" in model.feature_names:
        if AMOUNT_BUCKET_ARTIFACTS_PATH:
            artifacts_path = Path(AMOUNT_BUCKET_ARTIFACTS_PATH)
            if not artifacts_path.exists():
                raise SystemExit(
                    f"[startup] REFUSING TO START: AMOUNT_BUCKET_ARTIFACTS_PATH="
                    f"{artifacts_path} does not exist."
                )
            payload = json.loads(artifacts_path.read_text())
            inner_edges = payload["amount_bucket_inner_edges"]
            bin_edges = [float("-inf"), *[float(x) for x in inner_edges], float("inf")]
            configure_amount_bucket_artifacts(bin_edges, payload["amount_bucket_labels"])
            print(
                f"[startup] loaded amount_bucket artifacts from {artifacts_path} "
                f"(fit_from_train_cut_timestamp="
                f"{payload.get('fit_from_train_cut_timestamp')})",
                file=sys.stderr,
            )
        else:
            msg = (
                "Model requires 'amount_bucket' but AMOUNT_BUCKET_ARTIFACTS_PATH "
                "is not set. Run `python build_phase2_features.py --input "
                "<original training csv> --save-artifacts <path>.json` once, "
                "then set AMOUNT_BUCKET_ARTIFACTS_PATH to that file."
            )
            if ALLOW_UNKNOWN_FEATURES:
                print(f"[startup] WARNING (ALLOW_UNKNOWN_FEATURES=1): {msg}", file=sys.stderr)
            else:
                raise SystemExit(f"[startup] REFUSING TO START: {msg}")

    state.model = model
    state.class_order = class_order
    state.feature_names = model.feature_names
    state.model_path = str(model_path)
    state.loaded_at = datetime.now(timezone.utc)

    print(
        f"[startup] loaded model {model_path} "
        f"({len(model.feature_names)} features, class_order={class_order})",
        file=sys.stderr,
    )

    yield
    # ---- shutdown: nothing to clean up ----


app = FastAPI(
    title="Payment Retry Sequencer — Policy API",
    description=(
        "Phase 5 service: takes a failed payment-attempt event, runs it "
        "through the Phase 2 decline-reason classifier and the Phase 3 "
        "guardrailed policy engine, and returns a recommended retry "
        "action. risk_block and card_expired are hard-gated to no_retry "
        "inside the policy engine and cannot be overridden by this API."
    ),
    version="0.5.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PreviousAttempt(BaseModel):
    """One entry of `previous_attempts`, oldest first (schema.md)."""

    bank_response_code: Optional[str] = Field(
        None, description="Raw bank response code on that earlier attempt."
    )
    decline_reason_category: Optional[str] = Field(
        None, description="Classified decline reason on that earlier attempt, if known."
    )
    timestamp: Optional[datetime] = Field(
        None, description="When that earlier attempt occurred."
    )


class PaymentAttemptEvent(BaseModel):
    """A single failed payment attempt, matching schema.md, to be evaluated."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: str = Field(..., description="Unique ID for this specific attempt.")
    merchant_id: str = Field(..., description="Synthetic merchant identifier.")
    customer_id: str = Field(..., description="Hashed/synthetic customer identifier — never real PII.")
    amount: float = Field(..., gt=0, description="Transaction amount, whole rupees.")
    currency: str = Field("INR", description="Fixed to INR for this project's scope.")
    payment_method: Literal["card", "upi", "netbanking", "wallet"] = Field(
        ..., description="Payment method used for this attempt."
    )
    gateway: str = Field(..., description="Synthetic gateway name (e.g. gateway_a).")
    bank_response_code: str = Field(
        ..., description="Raw, noisy code as received from the bank — the signal the classifier interprets."
    )
    timestamp: datetime = Field(..., description="When this failed attempt occurred.")
    attempt_number: int = Field(..., ge=1, description="1 for the original attempt, incrementing per retry.")
    is_recurring: bool = Field(False, description="True for subscription/mandate payments.")
    previous_attempts: List[PreviousAttempt] = Field(
        default_factory=list,
        description=(
            "History of prior attempts for this transaction, oldest first. "
            "Used to derive attempt_count and last_attempt_time context "
            "unless the explicit fields below are given."
        ),
    )

    decline_reason_category: Optional[str] = Field(
        None,
        description=(
            "Present in the canonical schema as GROUND TRUTH, but not "
            "available at inference time and IGNORED by this endpoint. "
            "The Phase 2 classifier determines the decline reason from "
            "bank_response_code; do not rely on this field being used."
        ),
    )

    # --- explicit TransactionContext overrides; fall back to deriving from
    #     previous_attempts when not given ---
    attempt_count: Optional[int] = Field(
        None,
        description=(
            "Total attempts so far including this one. If omitted, derived "
            "as len(previous_attempts) + 1."
        ),
    )
    last_attempt_time: Optional[datetime] = Field(
        None,
        description=(
            "Timestamp of the prior attempt. If omitted, derived from "
            "previous_attempts[-1].timestamp when available."
        ),
    )
    customer_opted_out: bool = Field(
        False, description="True if the customer has opted out of automated retries."
    )
    gateway_recent_failure_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Recent failure rate for this gateway, if known. Used both by "
            "the policy engine's context and as a Phase 2 model feature — "
            "no safe fallback if the model needs it and this is omitted."
        ),
    )

    # --- explicit overrides for the second batch of model features (see
    #     feature_engineering.py docstring). Where omitted, the safe ones
    #     are derived from timestamp/attempt_number/previous_attempts; the
    #     cross-transaction stats below have no safe fallback and MUST be
    #     supplied or scoring will fail with a clear error. ---
    is_retry: Optional[bool] = Field(
        None, description="Whether this is a retry. If omitted, derived as attempt_number > 1."
    )
    hour: Optional[int] = Field(
        None, ge=0, le=23, description="Hour of day (0-23). If omitted, derived from timestamp."
    )
    is_weekend: Optional[bool] = Field(
        None, description="Whether timestamp falls on Sat/Sun. If omitted, derived from timestamp."
    )
    day_of_month: Optional[int] = Field(
        None, ge=1, le=31, description="Day of month (1-31). If omitted, derived from timestamp."
    )
    is_near_month_end: Optional[bool] = Field(
        None,
        description=(
            "Whether timestamp falls in the last 4 calendar days of the "
            "month (relevant to insufficient_funds timing). If omitted, "
            "derived from timestamp."
        ),
    )
    n_prev_attempts_this_txn: Optional[int] = Field(
        None,
        ge=0,
        description="Number of prior attempts for THIS transaction. If omitted, derived as len(previous_attempts).",
    )
    merchant_prior_attempt_count: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "This merchant's total prior attempt count across ALL their "
            "transactions. No safe fallback — required if the model needs it."
        ),
    )
    customer_hist_failure_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description=(
            "This customer's historical failure rate across ALL their "
            "transactions. No safe fallback — required if the model needs it."
        ),
    )
    customer_prior_attempt_count: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "This customer's total prior attempt count across ALL their "
            "transactions. No safe fallback — required if the model needs it."
        ),
    )
    merchant_hist_failure_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Merchant's historical failure rate, if known. Used both by "
            "the policy engine's context and as a Phase 2 model feature — "
            "no safe fallback if the model needs it and this is omitted."
        ),
    )


class EvaluateAttemptResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for this API call, also used as the audit-log key.")
    transaction_id: str = Field(..., description="Echoed from the request.")
    predicted_category: str = Field(..., description="Phase 2 classifier's predicted decline_reason_category.")
    predicted_confidence: float = Field(..., description="Phase 2 classifier's confidence in predicted_category.")
    recommended_action: str = Field(..., description="Phase 3 policy engine's recommended retry action.")
    timing: Optional[str] = Field(None, description="When to take the action (e.g. 'immediate', 'delay_300s'), or null for no_retry.")
    confidence: float = Field(..., description="Policy engine's confidence in recommended_action. Note: distinct from predicted_confidence.")
    reason_string: str = Field(..., description="Human-readable explanation of the decision, including any guardrail downgrades.")
    guardrail_checks: List[str] = Field(..., description="Every guardrail evaluated and its outcome, for audit purposes.")
    flagged_for_human_review: bool = Field(..., description="True if this decision requires a human to look at it.")
    evaluated_at: datetime = Field(..., description="When this evaluation was performed (UTC).")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(..., description="'ok' if the model is loaded and ready.")
    model_loaded: bool = Field(..., description="Whether the Phase 2 model is loaded in memory.")
    model_path: Optional[str] = Field(None, description="Path of the loaded model file.")
    model_feature_count: Optional[int] = Field(None, description="Number of features the loaded model expects.")
    loaded_at: Optional[datetime] = Field(None, description="When the model was loaded (UTC).")


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def write_audit_log(
    request_id: str,
    received_at: datetime,
    request_payload: dict,
    feature_row: dict,
    predicted_category: str,
    predicted_confidence: float,
    decision: dict,
) -> None:
    """
    Append one JSON line per evaluated attempt. Thread-locked file append —
    fine for a single-process local run; if you scale to multiple workers
    or processes, replace this with a proper queue/DB sink so records from
    different workers don't interleave badly, and so a slow disk doesn't
    block the request path.
    """
    record = {
        "request_id": request_id,
        "timestamp": received_at.isoformat(),
        "request": request_payload,
        "features_used": feature_row,
        "classifier_output": {
            "predicted_category": predicted_category,
            "predicted_confidence": predicted_confidence,
        },
        "policy_decision": decision,
    }
    line = json.dumps(record, default=str)
    with _audit_lock:
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/evaluate-attempt",
    response_model=EvaluateAttemptResponse,
    summary="Evaluate a failed payment attempt",
    description=(
        "Runs a failed payment-attempt event through the Phase 2 classifier "
        "and Phase 3 policy engine (hard rules -> ML/heuristic scoring -> "
        "downgrade-only guardrails) and returns the recommended retry "
        "action. Every call is appended to the local JSONL audit log."
    ),
)
async def evaluate_attempt(event: PaymentAttemptEvent) -> EvaluateAttemptResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    request_id = str(uuid.uuid4())
    received_at = datetime.now(timezone.utc)

    event_dict = event.model_dump()
    prev_list = [p.model_dump() for p in event.previous_attempts]

    try:
        feature_row = build_feature_row(event_dict, prev_list, state.feature_names)
    except NotImplementedError as e:
        # A model feature has no builder/fallback and wasn't supplied
        # explicitly — refuse loudly rather than silently guessing.
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        # A feature IS implemented but the input data doesn't fit its
        # expected shape (e.g. merchant_id not matching '<archetype>_<NN>').
        # That's a bad request, not a missing implementation.
        raise HTTPException(status_code=422, detail=str(e))

    X = pd.DataFrame([feature_row])
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    proba = state.model.predict_proba(X)[0]
    if len(proba) != len(state.class_order):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model produced {len(proba)} class probabilities but "
                f"class_order has {len(state.class_order)} entries "
                f"({state.class_order}). Set PHASE2_CLASS_ORDER correctly."
            ),
        )
    pred_idx = int(proba.argmax())
    predicted_category = state.class_order[pred_idx]
    predicted_confidence = float(proba[pred_idx])

    clf = ClassifierOutput(
        decline_reason_category=predicted_category,
        confidence=predicted_confidence,
        bank_response_code=event.bank_response_code,
    )

    attempt_count = (
        event.attempt_count if event.attempt_count is not None else len(event.previous_attempts) + 1
    )
    last_attempt_time = event.last_attempt_time
    if last_attempt_time is None and event.previous_attempts:
        last_attempt_time = event.previous_attempts[-1].timestamp

    ctx = TransactionContext(
        transaction_id=event.transaction_id,
        attempt_count=attempt_count,
        last_attempt_time=last_attempt_time,
        customer_opted_out=event.customer_opted_out,
        gateway_recent_failure_rate=event.gateway_recent_failure_rate,
        merchant_hist_failure_rate=event.merchant_hist_failure_rate,
        amount=event.amount,
        payment_method=event.payment_method,
        now=received_at,
    )

    decision = evaluate_retry_policy(clf, ctx)

    response = EvaluateAttemptResponse(
        request_id=request_id,
        transaction_id=event.transaction_id,
        predicted_category=predicted_category,
        predicted_confidence=round(predicted_confidence, 4),
        recommended_action=decision["recommended_action"],
        timing=decision["timing"],
        confidence=decision["confidence"],
        reason_string=decision["reason_string"],
        guardrail_checks=decision["guardrail_checks"],
        flagged_for_human_review=decision["flagged_for_human_review"],
        evaluated_at=received_at,
    )

    write_audit_log(
        request_id=request_id,
        received_at=received_at,
        request_payload=event_dict,
        feature_row=feature_row,
        predicted_category=predicted_category,
        predicted_confidence=predicted_confidence,
        decision=decision,
    )

    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns whether the Phase 2 model is loaded and the service is ready to evaluate attempts.",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        model_path=state.model_path,
        model_feature_count=len(state.feature_names) if state.feature_names else None,
        loaded_at=state.loaded_at,
    )
