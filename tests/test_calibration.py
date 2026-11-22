"""Unit tests for calibration and artifact bundling."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from appguard.calibration import CalibrationManager, train_and_bundle_model
from appguard.engine import ScoringEngine


def test_calibration_manager_percentiles():
    calibrator = CalibrationManager(target_percentile=95.0)
    scores = np.linspace(0.01, 1.0, 100)
    stats = calibrator.compute_calibration_statistics(scores)

    assert stats["sample_count"] == 100
    assert stats["target_percentile"] == 95.0
    assert 0.94 <= stats["calibrated_threshold"] <= 0.96
    assert "caveat_notice" in stats


def test_end_to_end_train_and_bundle(tmp_path):
    # Create small synthetic dataframe
    from appguard.data_generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=123)
    train_df, val_df, _ = gen.generate_datasets(n_train=100, n_val=40, n_test_normal=20, n_test_anomaly=10)

    res = train_and_bundle_model(
        train_df=train_df,
        val_df=val_df,
        output_dir=tmp_path,
        artifact_name="test_artifact.pt",
        epochs=3,
        batch_size=32,
        latent_dim=4,
        target_percentile=90.0,
    )

    art_path = res["artifact_path"]
    assert art_path.exists()
    assert res["metadata_path"].exists()

    # Load artifact through ScoringEngine
    engine = ScoringEngine(art_path)
    assert engine.threshold > 0
    assert engine.model is not None
    assert engine.preprocessor is not None

    sample_record = val_df.iloc[0].to_dict()
    score_res = engine.score_application(sample_record)
    assert score_res.anomaly_score >= 0
    assert isinstance(score_res.review_recommended, bool)
    assert len(score_res.top_deviations) > 0
