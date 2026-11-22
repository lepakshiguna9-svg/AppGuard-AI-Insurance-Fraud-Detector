"""Unit and integration tests for FastAPI AppGuard service."""

from fastapi.testclient import TestClient
import pytest
from appguard.api import app

client = TestClient(app)


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"


def test_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "calibrated_threshold" in data
        assert "regulatory_boundary" in data
        assert "synthetic_disclaimer" in data
        assert data["calibrated_threshold"] > 0


def test_score_single_valid_application():
    with TestClient(app) as client:
        payload = {
            "application_id": "TEST-APP-001",
            "policy_type": "auto",
            "coverage_amount": 50000.0,
            "deductible_amount": 500.0,
            "premium_frequency": "monthly",
            "application_channel": "online_direct",
            "applicant_age": 35.0,
            "annual_income_estimate": 70000.0,
            "prior_application_count": 0.0,
            "recent_address_changes_count": 0.0,
            "property_or_vehicle_age": 4.0,
            "beneficiary_relation": "spouse",
            "declared_credit_tier": "good",
        }
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["application_id"] == "TEST-APP-001"
        assert "anomaly_score" in data
        assert "calibrated_threshold" in data
        assert isinstance(data["review_recommended"], bool)
        assert isinstance(data["top_deviations"], list)
        assert "regulatory_notice" in data


def test_score_invalid_input_validation():
    with TestClient(app) as client:
        # Negative coverage amount and age < 18 must trigger 422
        payload = {
            "application_id": "TEST-APP-BAD",
            "policy_type": "auto",
            "coverage_amount": -500.0,
            "deductible_amount": 500.0,
            "premium_frequency": "monthly",
            "application_channel": "online_direct",
            "applicant_age": 14.0,  # below minimum age 18
            "annual_income_estimate": 70000.0,
        }
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422


def test_score_batch_with_partial_failures():
    with TestClient(app) as client:
        valid_rec = {
            "application_id": "REC-OK-01",
            "policy_type": "homeowners",
            "coverage_amount": 350000.0,
            "deductible_amount": 1000.0,
            "premium_frequency": "annual",
            "application_channel": "captive_agent",
            "applicant_age": 48.0,
            "annual_income_estimate": 110000.0,
            "prior_application_count": 0.0,
            "recent_address_changes_count": 0.0,
            "property_or_vehicle_age": 10.0,
            "beneficiary_relation": "spouse",
            "declared_credit_tier": "excellent",
        }
        malformed_rec = {
            "application_id": "REC-BAD-02",
            "policy_type": "invalid_policy_type_xyz",  # validation error
            "coverage_amount": -100.0,
        }

        payload = {"applications": [valid_rec, malformed_rec]}
        response = client.post("/api/v1/score/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["batch_size"] == 2
        assert data["successful_count"] == 1
        assert data["failed_count"] == 1

        results = data["results"]
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert "Validation or scoring error" in results[1]["error_message"]


def test_score_batch_size_limit():
    with TestClient(app) as client:
        # Exceeding 100 applications should fail validation
        large_batch = [{"policy_type": "auto", "coverage_amount": 10000.0} for _ in range(101)]
        response = client.post("/api/v1/score/batch", json={"applications": large_batch})
        assert response.status_code == 422
