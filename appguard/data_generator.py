"""Deterministic synthetic insurance application dataset generator.

Generates realistic tabular insurance applications without PII, protected demographics,
or sensitive identifiers. Provides reproducible random seeds for research and training.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from appguard.schema import (
    POLICY_TYPES,
    PREMIUM_FREQUENCIES,
    APPLICATION_CHANNELS,
    BENEFICIARY_RELATIONS,
    DECLARED_CREDIT_TIERS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


class SyntheticDataGenerator:
    """Generates synthetic insurance application datasets with realistic multivariate relationships."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _sample_typical_application(self, app_id: str) -> Dict[str, Any]:
        """Samples a single typical insurance application conforming to typical business distributions."""
        policy_type = self.rng.choice(
            POLICY_TYPES,
            p=[0.35, 0.25, 0.20, 0.15, 0.05]  # auto, home, term_life, renters, commercial
        )

        if policy_type == "auto":
            applicant_age = float(np.clip(self.rng.normal(38, 12), 18, 80))
            income = float(np.clip(self.rng.lognormal(11.0, 0.45), 25_000, 250_000))
            coverage = float(self.rng.choice([25_000, 50_000, 100_000, 250_000, 500_000], p=[0.2, 0.4, 0.25, 0.1, 0.05]))
            deductible = float(self.rng.choice([250, 500, 1000, 1500, 2000], p=[0.1, 0.5, 0.3, 0.07, 0.03]))
            vehicle_age = float(np.clip(self.rng.exponential(4.5), 0, 25))
            channel = self.rng.choice(APPLICATION_CHANNELS, p=[0.35, 0.20, 0.25, 0.10, 0.10])
            freq = self.rng.choice(PREMIUM_FREQUENCIES, p=[0.60, 0.10, 0.15, 0.15])
            beneficiary = self.rng.choice(BENEFICIARY_RELATIONS, p=[0.45, 0.35, 0.10, 0.08, 0.01, 0.01])
            credit = self.rng.choice(DECLARED_CREDIT_TIERS, p=[0.30, 0.45, 0.20, 0.05])
            prior_apps = float(self.rng.poisson(0.4))
            address_changes = float(self.rng.poisson(0.3))

        elif policy_type == "homeowners":
            applicant_age = float(np.clip(self.rng.normal(46, 11), 24, 85))
            income = float(np.clip(self.rng.lognormal(11.6, 0.5), 45_000, 450_000))
            coverage = float(np.clip(income * self.rng.uniform(2.5, 5.0), 150_000, 1_500_000))
            coverage = round(coverage / 10_000) * 10_000
            deductible = float(self.rng.choice([1000, 1500, 2000, 2500, 5000], p=[0.4, 0.25, 0.2, 0.1, 0.05]))
            vehicle_age = float(np.clip(self.rng.normal(18, 10), 0, 60))
            channel = self.rng.choice(APPLICATION_CHANNELS, p=[0.20, 0.35, 0.30, 0.10, 0.05])
            freq = self.rng.choice(PREMIUM_FREQUENCIES, p=[0.25, 0.10, 0.15, 0.50])
            beneficiary = self.rng.choice(BENEFICIARY_RELATIONS, p=[0.10, 0.65, 0.15, 0.08, 0.01, 0.01])
            credit = self.rng.choice(DECLARED_CREDIT_TIERS, p=[0.45, 0.40, 0.12, 0.03])
            prior_apps = float(self.rng.poisson(0.3))
            address_changes = float(self.rng.poisson(0.2))

        elif policy_type == "term_life":
            applicant_age = float(np.clip(self.rng.normal(39, 10), 21, 68))
            income = float(np.clip(self.rng.lognormal(11.4, 0.5), 35_000, 350_000))
            coverage = float(self.rng.choice([100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000], p=[0.1, 0.3, 0.35, 0.1, 0.1, 0.05]))
            deductible = 0.0
            vehicle_age = 0.0
            channel = self.rng.choice(APPLICATION_CHANNELS, p=[0.30, 0.30, 0.25, 0.10, 0.05])
            freq = self.rng.choice(PREMIUM_FREQUENCIES, p=[0.50, 0.10, 0.10, 0.30])
            beneficiary = self.rng.choice(BENEFICIARY_RELATIONS, p=[0.05, 0.65, 0.20, 0.08, 0.01, 0.01])
            credit = self.rng.choice(DECLARED_CREDIT_TIERS, p=[0.35, 0.45, 0.17, 0.03])
            prior_apps = float(self.rng.poisson(0.2))
            address_changes = float(self.rng.poisson(0.25))

        elif policy_type == "renters":
            applicant_age = float(np.clip(self.rng.normal(29, 7), 18, 65))
            income = float(np.clip(self.rng.lognormal(10.8, 0.4), 20_000, 150_000))
            coverage = float(self.rng.choice([15_000, 25_000, 35_000, 50_000, 75_000], p=[0.25, 0.40, 0.20, 0.10, 0.05]))
            deductible = float(self.rng.choice([250, 500, 1000], p=[0.35, 0.50, 0.15]))
            vehicle_age = float(np.clip(self.rng.normal(8, 5), 0, 30))
            channel = self.rng.choice(APPLICATION_CHANNELS, p=[0.55, 0.10, 0.10, 0.10, 0.15])
            freq = self.rng.choice(PREMIUM_FREQUENCIES, p=[0.75, 0.05, 0.05, 0.15])
            beneficiary = self.rng.choice(BENEFICIARY_RELATIONS, p=[0.60, 0.20, 0.05, 0.12, 0.01, 0.02])
            credit = self.rng.choice(DECLARED_CREDIT_TIERS, p=[0.20, 0.45, 0.25, 0.10])
            prior_apps = float(self.rng.poisson(0.5))
            address_changes = float(self.rng.poisson(0.6))

        else:  # commercial_property
            applicant_age = float(np.clip(self.rng.normal(49, 9), 25, 75))
            income = float(np.clip(self.rng.lognormal(12.2, 0.6), 75_000, 1_200_000))
            coverage = float(self.rng.choice([500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000], p=[0.25, 0.35, 0.25, 0.10, 0.05]))
            deductible = float(self.rng.choice([2500, 5000, 10000, 25000], p=[0.35, 0.40, 0.20, 0.05]))
            vehicle_age = float(np.clip(self.rng.normal(12, 8), 0, 45))
            channel = self.rng.choice(APPLICATION_CHANNELS, p=[0.10, 0.50, 0.35, 0.05, 0.0])
            freq = self.rng.choice(PREMIUM_FREQUENCIES, p=[0.20, 0.30, 0.10, 0.40])
            beneficiary = self.rng.choice(BENEFICIARY_RELATIONS, p=[0.05, 0.10, 0.05, 0.05, 0.65, 0.10])
            credit = self.rng.choice(DECLARED_CREDIT_TIERS, p=[0.50, 0.35, 0.12, 0.03])
            prior_apps = float(self.rng.poisson(0.6))
            address_changes = float(self.rng.poisson(0.2))

        return {
            "application_id": app_id,
            "policy_type": policy_type,
            "coverage_amount": round(float(coverage), 2),
            "deductible_amount": round(float(deductible), 2),
            "premium_frequency": freq,
            "application_channel": channel,
            "applicant_age": round(float(applicant_age), 1),
            "annual_income_estimate": round(float(income), 2),
            "prior_application_count": float(min(15, prior_apps)),
            "recent_address_changes_count": float(min(10, address_changes)),
            "property_or_vehicle_age": round(float(vehicle_age), 1),
            "beneficiary_relation": beneficiary,
            "declared_credit_tier": credit,
            "is_synthetic_anomaly": False,
            "anomaly_archetype": "none",
        }

    def _sample_unusual_application(self, app_id: str) -> Dict[str, Any]:
        """Synthesizes an intentionally unusual application for benchmarking anomaly detection."""
        archetype = self.rng.choice([
            "income_coverage_mismatch",
            "high_application_velocity",
            "channel_product_incongruity",
            "zero_deductible_max_limit",
            "youth_commercial_extreme",
        ])

        app = self._sample_typical_application(app_id)
        app["is_synthetic_anomaly"] = True
        app["anomaly_archetype"] = archetype

        if archetype == "income_coverage_mismatch":
            # Very low declared income with extraordinarily high requested coverage
            app["annual_income_estimate"] = float(round(self.rng.uniform(14_000, 24_000), 2))
            app["policy_type"] = "term_life"
            app["coverage_amount"] = float(round(self.rng.choice([1_500_000, 2_000_000, 3_000_000]), 2))
            app["deductible_amount"] = 0.0
            app["beneficiary_relation"] = "other"

        elif archetype == "high_application_velocity":
            # High rapid application velocity and frequent address jumps
            app["prior_application_count"] = float(self.rng.integers(7, 18))
            app["recent_address_changes_count"] = float(self.rng.integers(5, 12))
            app["declared_credit_tier"] = "limited_history"
            app["application_channel"] = "aggregator_partner"

        elif archetype == "channel_product_incongruity":
            # Multi-million commercial property submitted via unassisted mobile app
            app["policy_type"] = "commercial_property"
            app["coverage_amount"] = 4_500_000.0
            app["application_channel"] = "mobile_app"
            app["deductible_amount"] = 250.0  # unusually small deductible for $4.5M commercial policy
            app["applicant_age"] = 21.0

        elif archetype == "zero_deductible_max_limit":
            # Homeowners policy with $1.5M coverage and zero deductible, very frequent payment
            app["policy_type"] = "homeowners"
            app["coverage_amount"] = 1_850_000.0
            app["deductible_amount"] = 0.0
            app["premium_frequency"] = "monthly"
            app["recent_address_changes_count"] = 4.0

        elif archetype == "youth_commercial_extreme":
            app["policy_type"] = "commercial_property"
            app["applicant_age"] = 18.5
            app["annual_income_estimate"] = 18_000.0
            app["coverage_amount"] = 3_000_000.0
            app["prior_application_count"] = 6.0
            app["beneficiary_relation"] = "other"

        return app

    def generate_datasets(
        self,
        n_train: int = 5000,
        n_val: int = 1000,
        n_test_normal: int = 800,
        n_test_anomaly: int = 200,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates train, validation, and test splits with reproducible distributions."""
        # 1. Train set: strictly nominal/typical cases (unsupervised learning representation)
        train_rows = [
            self._sample_typical_application(f"APP-TRN-{i+1:05d}")
            for i in range(n_train)
        ]
        train_df = pd.DataFrame(train_rows)

        # 2. Validation / Calibration set: strictly typical cases for threshold calibration
        val_rows = [
            self._sample_typical_application(f"APP-VAL-{i+1:05d}")
            for i in range(n_val)
        ]
        val_df = pd.DataFrame(val_rows)

        # 3. Test set: mixed typical and synthetic unusual anomalies
        test_normal = [
            self._sample_typical_application(f"APP-TST-NORM-{i+1:04d}")
            for i in range(n_test_normal)
        ]
        test_anom = [
            self._sample_unusual_application(f"APP-TST-ANOM-{i+1:04d}")
            for i in range(n_test_anomaly)
        ]
        test_combined = test_normal + test_anom
        # Deterministically shuffle test set
        self.rng.shuffle(test_combined)
        test_df = pd.DataFrame(test_combined)

        return train_df, val_df, test_df


def save_synthetic_data(data_dir: Path, seed: int = 42) -> Dict[str, Path]:
    """Generates and writes CSV dataset splits to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticDataGenerator(seed=seed)
    train_df, val_df, test_df = generator.generate_datasets()

    paths = {
        "train": data_dir / "insurance_applications_train.csv",
        "val": data_dir / "insurance_applications_val.csv",
        "test": data_dir / "insurance_applications_test.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    # Save summary metadata
    summary = {
        "dataset_name": "AppGuard Synthetic Insurance Application Benchmark",
        "seed": seed,
        "splits": {
            "train": {"total": len(train_df), "anomalies": 0},
            "validation": {"total": len(val_df), "anomalies": 0},
            "test": {
                "total": len(test_df),
                "normal": int((test_df["is_synthetic_anomaly"] == False).sum()),
                "anomalies": int((test_df["is_synthetic_anomaly"] == True).sum()),
            },
        },
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "synthetic_notice": (
            "Educational synthetic data only. No real policyholder or applicant data is used. "
            "Excludes names, SSN, birth dates, street addresses, and protected attributes."
        ),
    }

    with open(data_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return paths


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data"
    created = save_synthetic_data(out_dir)
    print(f"Generated synthetic datasets in {out_dir}:")
    for k, v in created.items():
        print(f"  - {k}: {v.name}")
