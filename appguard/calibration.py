"""Model training, calibration, and artifact bundling for AppGuard AI.

Implements:
- Reproducible VAE model training with early stopping / loss logging.
- Empirical percentile-based calibration on a held-out nominal validation set.
- Versioned artifact packaging containing weights, preprocessor state, thresholds, and metadata.
- Clear documentation of calibration nuances and contamination caveats.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from appguard.schema import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from appguard.preprocessor import TabularPreprocessor
from appguard.model import TabularVAE


class CalibrationManager:
    """Manages empirical threshold calibration on held-out nominal applications."""

    def __init__(self, target_percentile: float = 95.0):
        self.target_percentile = target_percentile

    def compute_calibration_statistics(
        self,
        nominal_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates distribution metrics and empirical quantile thresholds."""
        scores = np.asarray(nominal_scores, dtype=np.float64)
        if len(scores) == 0:
            raise ValueError("Cannot calibrate on empty scores array.")

        p50 = float(np.percentile(scores, 50.0))
        p75 = float(np.percentile(scores, 75.0))
        p90 = float(np.percentile(scores, 90.0))
        p95 = float(np.percentile(scores, 95.0))
        p98 = float(np.percentile(scores, 98.0))
        p99 = float(np.percentile(scores, 99.0))
        primary_threshold = float(np.percentile(scores, self.target_percentile))

        return {
            "sample_count": len(scores),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "percentiles": {
                "p50": p50,
                "p75": p75,
                "p90": p90,
                "p95": p95,
                "p98": p98,
                "p99": p99,
            },
            "calibrated_threshold": primary_threshold,
            "target_percentile": self.target_percentile,
            "caveat_notice": (
                "Calibration was conducted on a held-out set intended to represent nominal applications. "
                "Any contamination in the calibration data shifts this empirical threshold. Legitimate "
                "unusual applications may exceed this threshold, while sophisticated fraudulent applications "
                "conforming to typical patterns will not be flagged. This threshold is an operational review "
                "triage guideline, NOT a proven boundary of fraud or moral culpability."
            ),
        }


def train_and_bundle_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    artifact_name: str = "appguard_v1.pt",
    latent_dim: int = 8,
    hidden_dims: Optional[List[int]] = None,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    beta: float = 0.01,
    seed: int = 42,
    target_percentile: float = 95.0,
) -> Dict[str, Any]:
    """Fits preprocessor, trains TabularVAE, calibrates threshold, and packages artifact."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)

    # PyTorch DataLoaders
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    # 2. Initialize VAE Model
    input_dim = preprocessor.total_encoded_dim
    hidden_dims = hidden_dims or [64, 32]
    model = TabularVAE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout_rate=0.05,
        beta=beta,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 3. Training Loop
    history: List[Dict[str, float]] = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []

        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch_x)
            loss, recon_l, kl_l = model.compute_loss(batch_x, x_recon, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_recon_losses.append(recon_l.item())
            epoch_kl_losses.append(kl_l.item())

        mean_loss = float(np.mean(epoch_losses))
        mean_recon = float(np.mean(epoch_recon_losses))
        mean_kl = float(np.mean(epoch_kl_losses))

        history.append({
            "epoch": epoch,
            "total_loss": mean_loss,
            "recon_loss": mean_recon,
            "kl_loss": mean_kl,
        })

    # 4. Calibration on Held-out Nominal Validation Set
    model.eval()
    with torch.no_grad():
        val_scores = model.score_samples(val_tensor).cpu().numpy()

    calibrator = CalibrationManager(target_percentile=target_percentile)
    calibration_stats = calibrator.compute_calibration_statistics(val_scores)

    # 5. Package Artifact Bundle
    artifact_bundle = {
        "artifact_version": "1.0.0",
        "model_architecture": "TabularVAE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "feature_schema": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "total_encoded_dimension": input_dim,
        },
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
        "preprocessor_state": preprocessor.to_dict(),
        "calibration": calibration_stats,
        "training_metadata": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "final_epoch_loss": history[-1] if history else {},
            "history_summary": {
                "initial_total_loss": history[0]["total_loss"] if history else None,
                "final_total_loss": history[-1]["total_loss"] if history else None,
            },
        },
        "regulatory_disclaimer": (
            "AppGuard AI anomaly detection models generate probabilistic reconstruction error signals reflecting "
            "statistical atypicality relative to nominal baseline distributions. Scores do not prove fraud or "
            "misrepresentation. In production insurance operations, applicable legal, regulatory, and adverse-action "
            "requirements vary significantly by jurisdiction, product line, and intended use; any operational "
            "deployment requires formal review by qualified legal and insurance compliance professionals. Within "
            "this system, scores serve solely as an operational review signal to prioritize human underwriting triage, "
            "and must never be used for automated approval, denial, pricing, cancellation, or other adverse actions. "
            "This educational demonstration does not provide legal or regulatory advice."
        ),
    }

    artifact_path = output_dir / artifact_name
    torch.save(artifact_bundle, artifact_path)

    # Also save human-readable JSON metadata
    metadata_json_path = output_dir / "model_metadata.json"
    metadata_export = {
        k: v for k, v in artifact_bundle.items() if k not in ("state_dict", "preprocessor_state")
    }
    with open(metadata_json_path, "w") as f:
        json.dump(metadata_export, f, indent=2)

    return {
        "artifact_path": artifact_path,
        "metadata_path": metadata_json_path,
        "calibrated_threshold": calibration_stats["calibrated_threshold"],
        "calibration_stats": calibration_stats,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_file = base_dir / "data" / "insurance_applications_train.csv"
    val_file = base_dir / "data" / "insurance_applications_val.csv"
    art_dir = base_dir / "artifacts"

    if not train_file.exists() or not val_file.exists():
        from appguard.data_generator import save_synthetic_data
        save_synthetic_data(base_dir / "data")

    tr_df = pd.read_csv(train_file)
    vl_df = pd.read_csv(val_file)

    res = train_and_bundle_model(
        train_df=tr_df,
        val_df=vl_df,
        output_dir=art_dir,
        epochs=35,
        batch_size=64,
        latent_dim=8,
    )
    print("Training and calibration complete.")
    print(f"Artifact saved to: {res['artifact_path']}")
    print(f"Calibrated 95th-percentile threshold: {res['calibrated_threshold']:.5f}")
