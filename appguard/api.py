"""FastAPI service for AppGuard AI Insurance Application Fraud Detector.

Endpoints:
- GET  /api/v1/health
- GET  /api/v1/model-info
- POST /api/v1/score
- POST /api/v1/score/batch
- GET  /api/v1/demo/samples
- POST /api/v1/demo/generate-batch
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from appguard.schema import (
    InsuranceApplicationInput,
    ApplicationScoreResult,
    POLICY_TYPES,
    PREMIUM_FREQUENCIES,
    APPLICATION_CHANNELS,
    BENEFICIARY_RELATIONS,
    DECLARED_CREDIT_TIERS,
)
from appguard.engine import ScoringEngine
from appguard.data_generator import SyntheticDataGenerator

# Configure safe structured logging (Never log PII or raw application field payloads)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [appguard.api] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("appguard.api")

DEFAULT_ARTIFACT_PATH = Path(
    os.getenv("APPGUARD_ARTIFACT_PATH", Path(__file__).resolve().parent.parent / "artifacts" / "appguard_v1.pt")
)

# Global engine state
engine: Optional[ScoringEngine] = None
start_time_epoch: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes scoring engine during service startup."""
    global engine
    logger.info("Initializing AppGuard scoring engine...")
    if not DEFAULT_ARTIFACT_PATH.exists():
        logger.warning(
            f"Artifact bundle not found at {DEFAULT_ARTIFACT_PATH}. "
            f"Attempting automatic generation..."
        )
        from appguard.data_generator import save_synthetic_data
        from appguard.calibration import train_and_bundle_model
        import pandas as pd

        base_dir = DEFAULT_ARTIFACT_PATH.parent.parent
        data_dir = base_dir / "data"
        train_file = data_dir / "insurance_applications_train.csv"
        val_file = data_dir / "insurance_applications_val.csv"

        if not train_file.exists() or not val_file.exists():
            save_synthetic_data(data_dir)

        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        train_and_bundle_model(train_df, val_df, DEFAULT_ARTIFACT_PATH.parent)

    engine = ScoringEngine(DEFAULT_ARTIFACT_PATH)
    logger.info(
        f"AppGuard engine initialized successfully. "
        f"Version: {engine.model_metadata.get('artifact_version')}, "
        f"Threshold: {engine.threshold:.5f}"
    )
    yield
    logger.info("Shutting down AppGuard scoring engine.")


app = FastAPI(
    title="AppGuard AI: Insurance Application Fraud Detector API",
    description=(
        "Unsupervised anomaly detection scoring API for insurance application triage. "
        "Flags statistically unusual submissions for operational human review. "
        "Does NOT constitute proof of fraud or provide automated adverse decisioning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local workbench frontend development and container networking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response schemas
class BatchScoreRequest(BaseModel):
    """Request model for batch application scoring."""
    applications: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of raw application records to score (max 100 per request)."
    )


class BatchScoreRecordResponse(BaseModel):
    """Result for an individual application within a batch response."""
    record_index: int
    status: str
    application_id: Optional[str] = None
    anomaly_score: Optional[float] = None
    calibrated_threshold: Optional[float] = None
    review_recommended: Optional[bool] = None
    review_priority: Optional[str] = None
    neutral_summary: Optional[str] = None
    regulatory_notice: Optional[str] = None
    top_deviations: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None


class BatchScoreResponse(BaseModel):
    """Container for batch scoring execution summary and per-record outcomes."""
    batch_size: int
    successful_count: int
    failed_count: int
    results: List[BatchScoreRecordResponse]
    regulatory_notice: str


class GenerateDemoBatchRequest(BaseModel):
    count: int = Field(default=15, ge=1, le=100)
    include_anomalies: bool = Field(default=True)


# Middlewares for structured audit metrics (without leaking request body)
@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    start_t = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    # Avoid logging payload contents; record only status and duration
    if request.url.path.startswith("/api/"):
        logger.info(
            f"METHOD={request.method} PATH={request.url.path} "
            f"STATUS={response.status_code} DURATION_MS={elapsed_ms:.2f}"
        )
    return response


@app.get("/api/v1/health", tags=["System"])
def get_health():
    """Health check endpoint reporting service status and model availability."""
    is_ready = engine is not None and engine.model is not None
    uptime_sec = round(time.time() - start_time_epoch, 2)
    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "AppGuard AI Fraud Detector",
        "version": "1.0.0",
        "model_loaded": is_ready,
        "uptime_seconds": uptime_sec,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/model-info", tags=["Model & Governance"])
def get_model_info():
    """Returns governance metadata, architecture parameters, and calibrated threshold."""
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine is not initialized.",
        )
    return engine.get_info()


@app.post("/api/v1/score", tags=["Scoring"])
def score_single_application(application: InsuranceApplicationInput):
    """Scores a single insurance application.

    Validates input schema, computes VAE reconstruction error, evaluates against
    the calibrated threshold, identifies largest feature-level statistical deviations,
    and returns a neutral operational recommendation.
    """
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine is not initialized.",
        )

    try:
        result: ApplicationScoreResult = engine.score_application(application)
        return {
            "status": "success",
            "application_id": result.application_id,
            "anomaly_score": result.anomaly_score,
            "calibrated_threshold": result.calibrated_threshold,
            "review_recommended": result.review_recommended,
            "review_priority": result.review_priority,
            "top_deviations": [
                {
                    "feature_name": d.feature_name,
                    "feature_type": d.feature_type,
                    "submitted_value": d.submitted_value,
                    "reconstructed_value": d.reconstructed_value,
                    "reconstruction_error": d.reconstruction_error,
                    "statistical_observation": d.statistical_observation,
                }
                for d in result.top_deviations
            ],
            "neutral_summary": result.neutral_summary,
            "regulatory_notice": result.regulatory_notice,
        }
    except Exception as e:
        logger.error(f"Error scoring application: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failure: {str(e)}",
        )


@app.post("/api/v1/score/batch", tags=["Scoring"], response_model=BatchScoreResponse)
def score_batch_applications(payload: BatchScoreRequest):
    """Scores a batch of up to 100 insurance applications.

    Provides safe per-record error isolation: malformed records return a clean error message
    without failing the entire batch or exposing stack traces.
    """
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine is not initialized.",
        )

    records = payload.applications
    raw_results = engine.score_batch(records, max_batch_size=100)

    success_cnt = sum(1 for r in raw_results if r["status"] == "success")
    fail_cnt = sum(1 for r in raw_results if r["status"] == "error")

    formatted_results = [
        BatchScoreRecordResponse(
            record_index=r["record_index"],
            status=r["status"],
            application_id=r.get("application_id"),
            anomaly_score=r.get("anomaly_score"),
            calibrated_threshold=r.get("calibrated_threshold"),
            review_recommended=r.get("review_recommended"),
            review_priority=r.get("review_priority"),
            neutral_summary=r.get("neutral_summary"),
            regulatory_notice=r.get("regulatory_notice"),
            top_deviations=r.get("top_deviations"),
            error_message=r.get("error_message"),
        )
        for r in raw_results
    ]

    return BatchScoreResponse(
        batch_size=len(records),
        successful_count=success_cnt,
        failed_count=fail_cnt,
        results=formatted_results,
        regulatory_notice=(
            "Batch scores reflect empirical statistical divergence from nominal baselines. "
            "Applicable requirements vary by jurisdiction and line of business; scores must not be used "
            "for automated adverse decisions, approval, or pricing. Applications flagged for review "
            "require qualified human underwriting evaluation."
        ),
    )


@app.get("/api/v1/demo/samples", tags=["Demo Mode"])
def get_demo_samples():
    """Returns curated synthetic sample applications for interactive workbench evaluation."""
    samples = [
        {
            "application_id": "DEMO-NORM-001",
            "policy_type": "auto",
            "coverage_amount": 50000.0,
            "deductible_amount": 500.0,
            "premium_frequency": "monthly",
            "application_channel": "online_direct",
            "applicant_age": 36.0,
            "annual_income_estimate": 78000.0,
            "prior_application_count": 0.0,
            "recent_address_changes_count": 0.0,
            "property_or_vehicle_age": 4.0,
            "beneficiary_relation": "spouse",
            "declared_credit_tier": "good",
            "description": "Standard suburban auto policy with proportional income and stable residence.",
        },
        {
            "application_id": "DEMO-NORM-002",
            "policy_type": "homeowners",
            "coverage_amount": 420000.0,
            "deductible_amount": 1500.0,
            "premium_frequency": "annual",
            "application_channel": "captive_agent",
            "applicant_age": 52.0,
            "annual_income_estimate": 140000.0,
            "prior_application_count": 1.0,
            "recent_address_changes_count": 0.0,
            "property_or_vehicle_age": 14.0,
            "beneficiary_relation": "spouse",
            "declared_credit_tier": "excellent",
            "description": "Typical established homeowner policy placed through traditional agent.",
        },
        {
            "application_id": "DEMO-ANOM-001",
            "policy_type": "term_life",
            "coverage_amount": 2500000.0,
            "deductible_amount": 0.0,
            "premium_frequency": "monthly",
            "application_channel": "online_direct",
            "applicant_age": 22.0,
            "annual_income_estimate": 19000.0,
            "prior_application_count": 0.0,
            "recent_address_changes_count": 1.0,
            "property_or_vehicle_age": 0.0,
            "beneficiary_relation": "other",
            "declared_credit_tier": "limited_history",
            "description": "Statistical mismatch: $2.5M life coverage with $19k income and unrelated beneficiary.",
        },
        {
            "application_id": "DEMO-ANOM-002",
            "policy_type": "auto",
            "coverage_amount": 100000.0,
            "deductible_amount": 250.0,
            "premium_frequency": "monthly",
            "application_channel": "aggregator_partner",
            "applicant_age": 31.0,
            "annual_income_estimate": 62000.0,
            "prior_application_count": 11.0,
            "recent_address_changes_count": 6.0,
            "property_or_vehicle_age": 3.0,
            "beneficiary_relation": "self",
            "declared_credit_tier": "limited_history",
            "description": "High velocity anomaly: 11 prior applications in 3 years with 6 address changes.",
        },
        {
            "application_id": "DEMO-ANOM-003",
            "policy_type": "commercial_property",
            "coverage_amount": 4000000.0,
            "deductible_amount": 500.0,
            "premium_frequency": "monthly",
            "application_channel": "mobile_app",
            "applicant_age": 19.0,
            "annual_income_estimate": 22000.0,
            "prior_application_count": 5.0,
            "recent_address_changes_count": 3.0,
            "property_or_vehicle_age": 2.0,
            "beneficiary_relation": "other",
            "declared_credit_tier": "fair",
            "description": "Channel & scale incongruity: $4M commercial property via mobile with teen applicant.",
        },
    ]
    return {
        "count": len(samples),
        "samples": samples,
        "disclaimer": "Curated synthetic scenarios designed for interactive operations testing.",
    }


@app.post("/api/v1/demo/generate-batch", tags=["Demo Mode"])
def generate_synthetic_batch(payload: GenerateDemoBatchRequest):
    """Generates a fresh batch of synthetic applications and scores them immediately."""
    gen = SyntheticDataGenerator(seed=int(time.time()) % 100_000)
    records: List[Dict[str, Any]] = []

    anomaly_cnt = max(1, int(payload.count * 0.25)) if payload.include_anomalies else 0
    normal_cnt = payload.count - anomaly_cnt

    for i in range(normal_cnt):
        records.append(gen._sample_typical_application(f"BATCH-NORM-{i+1:03d}"))
    for i in range(anomaly_cnt):
        records.append(gen._sample_unusual_application(f"BATCH-ANOM-{i+1:03d}"))

    gen.rng.shuffle(records)

    # Score batch through engine
    batch_res = engine.score_batch(records)
    return {
        "count": len(records),
        "scored_applications": batch_res,
        "raw_applications": records,
    }


# Mount production frontend static files if present
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

