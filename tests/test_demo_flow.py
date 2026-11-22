"""Integration tests for demo flow endpoints."""

from fastapi.testclient import TestClient
from appguard.api import app


def test_demo_samples_flow():
    with TestClient(app) as client:
        res = client.get("/api/v1/demo/samples")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] > 0
        samples = data["samples"]
        assert len(samples) >= 3

        # Test scoring each demo sample
        for sample in samples:
            score_res = client.post("/api/v1/score", json=sample)
            assert score_res.status_code == 200
            score_data = score_res.json()
            assert score_data["anomaly_score"] >= 0
            assert "review_recommended" in score_data


def test_demo_batch_generation_flow():
    with TestClient(app) as client:
        payload = {"count": 10, "include_anomalies": True}
        res = client.post("/api/v1/demo/generate-batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 10
        assert len(data["scored_applications"]) == 10
        assert len(data["raw_applications"]) == 10

        # Verify review recommendations exist
        review_flags = [item["review_recommended"] for item in data["scored_applications"] if item["status"] == "success"]
        assert len(review_flags) == 10
