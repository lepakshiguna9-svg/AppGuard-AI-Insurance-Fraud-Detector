"""Inference engine for AppGuard AI scoring and statistical attribution.

Loads versioned model artifacts and calculates:
- Sample-level reconstruction anomaly score.
- Review triage recommendation based on empirical threshold.
- Human-interpretable feature-level deviations (statistical discrepancies, never fraud accusations).
- Safe, neutral operational summaries and persistent regulatory notices.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch

from appguard.schema import (
    InsuranceApplicationInput,
    FeatureDeviation,
    ApplicationScoreResult,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from appguard.preprocessor import TabularPreprocessor
from appguard.model import TabularVAE


class ScoringEngine:
    """Production scoring engine loading versioned PyTorch VAE artifacts."""

    def __init__(self, artifact_path: Union[str, Path]):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Artifact bundle not found at {self.artifact_path}")

        self.bundle: Dict[str, Any] = {}
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.model: Optional[TabularVAE] = None
        self.threshold: float = 0.0
        self.calibration_stats: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}

        self._load_artifact()

    def _load_artifact(self) -> None:
        """Loads and verifies versioned artifact components."""
        # Use weights_only=False because bundle contains dict of metadata, state_dict, and preprocessor state
        bundle = torch.save if False else torch.load(self.artifact_path, map_location="cpu", weights_only=False)
        self.bundle = bundle

        version = bundle.get("artifact_version")
        if not version or not version.startswith("1."):
            raise ValueError(f"Incompatible artifact version '{version}'. Expected 1.x.x")

        # Reconstruct preprocessor
        self.preprocessor = TabularPreprocessor.from_dict(bundle["preprocessor_state"])

        # Reconstruct model
        model_cfg = bundle["model_config"]
        self.model = TabularVAE(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            latent_dim=model_cfg["latent_dim"],
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            beta=model_cfg.get("beta", 0.01),
        )
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

        # Calibration
        calib = bundle.get("calibration", {})
        self.threshold = float(calib.get("calibrated_threshold", 0.15))
        self.calibration_stats = calib

        self.model_metadata = {
            "artifact_version": bundle.get("artifact_version"),
            "model_architecture": bundle.get("model_architecture"),
            "created_at_utc": bundle.get("created_at_utc"),
            "calibrated_threshold": self.threshold,
            "target_percentile": calib.get("target_percentile", 95.0),
            "input_dimension": model_cfg["input_dim"],
            "latent_dimension": model_cfg["latent_dim"],
            "hidden_dims": model_cfg["hidden_dims"],
        }

    def get_info(self) -> Dict[str, Any]:
        """Returns non-sensitive metadata regarding active model and calibration."""
        return {
            **self.model_metadata,
            "regulatory_boundary": (
                "Anomaly detection scores represent statistical distance from nominal application profiles. "
                "They do NOT establish fraud, dishonesty, or elevated claim probability. "
                "Applicable legal and regulatory requirements vary by jurisdiction and use case; "
                "real-world deployments require review by qualified legal and compliance professionals. "
                "Within this system, scores are triage signals only and must not be used for automated "
                "approval, denial, pricing, cancellation, or other adverse decisions."
            ),
            "synthetic_disclaimer": (
                "Educational demo model trained on synthetic policy applications. "
                "Synthetic performance metrics must not be misrepresented as real-world underwriting outcomes."
            ),
        }

    def score_application(
        self,
        app_input: Union[InsuranceApplicationInput, Dict[str, Any]],
        top_k_deviations: int = 4,
    ) -> ApplicationScoreResult:
        """Scores a single policy application, decomposing reconstruction error into top deviations."""
        if isinstance(app_input, InsuranceApplicationInput):
            record = app_input.model_dump()
            app_id = app_input.application_id
        else:
            record = dict(app_input)
            app_id = record.get("application_id")

        if not self.preprocessor or not self.model:
            raise RuntimeError("Scoring engine is not properly initialized.")

        # Transform to normalized tensor
        x_np = self.preprocessor.transform([record])
        x_tensor = torch.tensor(x_np, dtype=torch.float32)

        with torch.no_grad():
            mu, _ = self.model.encode(x_tensor)
            x_recon_tensor = self.model.decode(mu)

        x_recon_np = x_recon_tensor.cpu().numpy()

        # Total sample anomaly score (Mean Squared Error across dimensions)
        diff = x_np[0] - x_recon_np[0]
        total_anomaly_score = float(np.mean(diff ** 2))

        # Determine review triage
        review_recommended = bool(total_anomaly_score > self.threshold)
        if not review_recommended:
            review_priority = "standard"
        elif total_anomaly_score > self.threshold * 1.6:
            review_priority = "high"
        else:
            review_priority = "elevated"

        # Feature-level reconstruction deviation attribution
        deviations: List[FeatureDeviation] = []

        for col, sl in self.preprocessor.column_slices.items():
            orig_val = record.get(col, None)
            slice_x = x_np[0, sl.start_idx:sl.end_idx]
            slice_recon = x_recon_np[0, sl.start_idx:sl.end_idx]
            feat_err = float(np.mean((slice_x - slice_recon) ** 2))

            if sl.feature_type == "numerical":
                recon_decoded = self.preprocessor.decode_numerical(col, float(slice_recon[0]))
                if col in ("coverage_amount", "annual_income_estimate", "deductible_amount"):
                    obs = (
                        f"Submitted ${orig_val:,.0f} vs typical reconstructed ${recon_decoded:,.0f} "
                        f"for this profile segment."
                    )
                elif col in ("prior_application_count", "recent_address_changes_count"):
                    obs = (
                        f"Submitted count of {orig_val:.0f} vs expected ~{recon_decoded:.1f} "
                        f"typical application frequency."
                    )
                else:
                    obs = f"Submitted {orig_val:.1f} vs expected {recon_decoded:.1f}."

                deviations.append(FeatureDeviation(
                    feature_name=col,
                    feature_type="numerical",
                    submitted_value=orig_val,
                    reconstructed_value=round(recon_decoded, 2),
                    reconstruction_error=round(feat_err, 4),
                    deviation_percentile=0.0,  # relative ranking
                    statistical_observation=obs,
                ))

            else:  # categorical
                recon_cat = self.preprocessor.decode_categorical(col, slice_recon)
                obs = (
                    f"Selected category '{orig_val}' shows low mutual association with other fields; "
                    f"nominal pattern suggests '{recon_cat}'."
                )
                deviations.append(FeatureDeviation(
                    feature_name=col,
                    feature_type="categorical",
                    submitted_value=orig_val,
                    reconstructed_value=recon_cat,
                    reconstruction_error=round(feat_err, 4),
                    deviation_percentile=0.0,
                    statistical_observation=obs,
                ))

        # Sort deviations descending by reconstruction error
        deviations.sort(key=lambda d: d.reconstruction_error, reverse=True)
        top_deviations = deviations[:top_k_deviations]

        # Neutral summary construction
        if review_recommended:
            top_feat_names = [d.feature_name.replace("_", " ") for d in top_deviations[:2]]
            neutral_summary = (
                f"Application reconstruction score ({total_anomaly_score:.3f}) exceeds the calibrated nominal "
                f"threshold ({self.threshold:.3f}). Most significant statistical variances observed in "
                f"{' and '.join(top_feat_names)}. Operational manual triage recommended."
            )
        else:
            neutral_summary = (
                f"Application reconstruction score ({total_anomaly_score:.3f}) aligns with nominal "
                f"distribution baselines (threshold: {self.threshold:.3f}). Standard underwriting routing."
            )

        regulatory_notice = (
            "NOTICE: AppGuard anomaly detection is an operational review triage indicator only and does not "
            "constitute evidence of fraud. Applicable legal and regulatory requirements vary by jurisdiction and "
            "product line; scores must not be used for automated approval, denial, pricing, cancellation, or "
            "other adverse actions without independent human underwriting adjudication."
        )

        return ApplicationScoreResult(
            application_id=app_id,
            anomaly_score=round(total_anomaly_score, 5),
            calibrated_threshold=round(self.threshold, 5),
            review_recommended=review_recommended,
            review_priority=review_priority,
            top_deviations=top_deviations,
            neutral_summary=neutral_summary,
            regulatory_notice=regulatory_notice,
        )

    def score_batch(
        self,
        applications: List[Dict[str, Any]],
        max_batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Scores a batch of applications with safe per-record error isolation."""
        if len(applications) > max_batch_size:
            raise ValueError(
                f"Batch size {len(applications)} exceeds maximum allowed limit of {max_batch_size} records."
            )

        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(applications):
            try:
                # Validate input using Pydantic
                validated_item = InsuranceApplicationInput(**item)
                res = self.score_application(validated_item)
                results.append({
                    "record_index": idx,
                    "status": "success",
                    "application_id": res.application_id,
                    "anomaly_score": res.anomaly_score,
                    "calibrated_threshold": res.calibrated_threshold,
                    "review_recommended": res.review_recommended,
                    "review_priority": res.review_priority,
                    "neutral_summary": res.neutral_summary,
                    "regulatory_notice": res.regulatory_notice,
                    "top_deviations": [
                        {
                            "feature_name": d.feature_name,
                            "feature_type": d.feature_type,
                            "submitted_value": d.submitted_value,
                            "reconstructed_value": d.reconstructed_value,
                            "reconstruction_error": d.reconstruction_error,
                            "statistical_observation": d.statistical_observation,
                        }
                        for d in res.top_deviations
                    ],
                })
            except Exception as e:
                # Safe per-record error formatting without leaking stack traces or PII
                app_id = item.get("application_id") if isinstance(item, dict) else None
                results.append({
                    "record_index": idx,
                    "status": "error",
                    "application_id": app_id,
                    "error_message": f"Validation or scoring error: {str(e)}",
                })

        return results
