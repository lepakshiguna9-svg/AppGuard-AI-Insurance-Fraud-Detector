"""Tabular preprocessor for mixed numeric and categorical insurance application data.

Features:
- Robust imputation for missing values (median for numeric, most frequent / explicit missing for categorical).
- Standardization for numerical features (mean/std or robust median/IQR).
- One-hot encoding for categorical features with zero-vector handling for unknown/unseen categories.
- Strict feature column mapping tracking so reconstruction error can be mapped back
  to original input fields for interpretable attribution.
- Full serialization dictionary for self-contained artifact bundling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

from appguard.schema import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    POLICY_TYPES,
    PREMIUM_FREQUENCIES,
    APPLICATION_CHANNELS,
    BENEFICIARY_RELATIONS,
    DECLARED_CREDIT_TIERS,
)


@dataclass
class ColumnSlice:
    """Tracks index range in the encoded feature tensor for an original feature."""
    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    start_idx: int
    end_idx: int  # exclusive
    categories: Optional[List[str]] = None


class TabularPreprocessor:
    """Preprocessor for mixed tabular insurance application features."""

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.numerical_features = numerical_features or list(NUMERICAL_FEATURES)
        self.categorical_features = categorical_features or list(CATEGORICAL_FEATURES)

        # Fitted statistics
        self.num_means_: Dict[str, float] = {}
        self.num_stds_: Dict[str, float] = {}
        self.num_medians_: Dict[str, float] = {}
        self.cat_categories_: Dict[str, List[str]] = {}
        self.cat_modes_: Dict[str, str] = {}

        # Column layout tracking for tensor representation
        self.column_slices: Dict[str, ColumnSlice] = {}
        self.total_encoded_dim: int = 0
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fits numerical scalers and categorical dictionaries on training data."""
        # 1. Fit numerical features
        for col in self.numerical_features:
            series = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series([0.0])
            median = float(series.median() if not series.dropna().empty else 0.0)
            mean = float(series.mean() if not series.dropna().empty else 0.0)
            std = float(series.std() if not series.dropna().empty and series.std() > 1e-6 else 1.0)

            self.num_medians_[col] = median
            self.num_means_[col] = mean
            self.num_stds_[col] = std

        # 2. Fit categorical features
        # We ensure standard known schema values are included so encoding is robust
        known_defaults = {
            "policy_type": POLICY_TYPES,
            "premium_frequency": PREMIUM_FREQUENCIES,
            "application_channel": APPLICATION_CHANNELS,
            "beneficiary_relation": BENEFICIARY_RELATIONS,
            "declared_credit_tier": DECLARED_CREDIT_TIERS,
        }

        for col in self.categorical_features:
            if col in df:
                observed = [str(x).strip().lower() for x in df[col].dropna().unique() if str(x).strip()]
                mode_val = str(df[col].mode().iloc[0]).strip().lower() if not df[col].dropna().empty else "unknown"
            else:
                observed = []
                mode_val = "unknown"

            defaults = known_defaults.get(col, [])
            combined = sorted(list(set(defaults + observed)))
            self.cat_categories_[col] = combined
            self.cat_modes_[col] = mode_val if mode_val in combined else (combined[0] if combined else "unknown")

        # 3. Construct column index layout
        current_idx = 0
        self.column_slices.clear()

        # Numerical first
        for col in self.numerical_features:
            self.column_slices[col] = ColumnSlice(
                feature_name=col,
                feature_type="numerical",
                start_idx=current_idx,
                end_idx=current_idx + 1,
            )
            current_idx += 1

        # Categorical one-hot encoded slices
        for col in self.categorical_features:
            cats = self.cat_categories_[col]
            dim = len(cats)
            self.column_slices[col] = ColumnSlice(
                feature_name=col,
                feature_type="categorical",
                start_idx=current_idx,
                end_idx=current_idx + dim,
                categories=cats,
            )
            current_idx += dim

        self.total_encoded_dim = current_idx
        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> np.ndarray:
        """Transforms input data into a standardized 2D float32 numpy tensor."""
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform.")

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        n_rows = len(df)
        encoded = np.zeros((n_rows, self.total_encoded_dim), dtype=np.float32)

        # Transform numerical features
        for col in self.numerical_features:
            sl = self.column_slices[col]
            mean = self.num_means_[col]
            std = self.num_stds_[col]
            median = self.num_medians_[col]

            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").fillna(median).to_numpy()
            else:
                series = np.full(n_rows, median, dtype=np.float32)

            # Standardize: (x - mean) / std
            standardized = (series - mean) / std
            encoded[:, sl.start_idx] = standardized.astype(np.float32)

        # Transform categorical features with one-hot encoding
        for col in self.categorical_features:
            sl = self.column_slices[col]
            cats = sl.categories or []
            cat_to_subidx = {cat: i for i, cat in enumerate(cats)}
            mode_val = self.cat_modes_[col]

            if col in df.columns:
                series = df[col].fillna(mode_val).astype(str).str.strip().str.lower()
            else:
                series = pd.Series([mode_val] * n_rows)

            for row_idx, val in enumerate(series):
                if val in cat_to_subidx:
                    sub_idx = cat_to_subidx[val]
                    encoded[row_idx, sl.start_idx + sub_idx] = 1.0
                # Unknown categories remain zero vector (safe graceful degradation)

        return encoded

    def decode_numerical(self, feature_name: str, scaled_value: float) -> float:
        """Inverts standardization for a single numerical feature value."""
        if feature_name not in self.num_means_:
            return scaled_value
        mean = self.num_means_[feature_name]
        std = self.num_stds_[feature_name]
        return float((scaled_value * std) + mean)

    def decode_categorical(self, feature_name: str, probs_or_logits: np.ndarray) -> str:
        """Finds most probable category from reconstructed vector slice."""
        sl = self.column_slices.get(feature_name)
        if not sl or not sl.categories:
            return "unknown"
        best_idx = int(np.argmax(probs_or_logits))
        if 0 <= best_idx < len(sl.categories):
            return sl.categories[best_idx]
        return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes preprocessor state for artifact storage."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted preprocessor.")
        return {
            "version": "1.0.0",
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "num_means": self.num_means_,
            "num_stds": self.num_stds_,
            "num_medians": self.num_medians_,
            "cat_categories": self.cat_categories_,
            "cat_modes": self.cat_modes_,
            "total_encoded_dim": self.total_encoded_dim,
            "column_slices": {
                k: {
                    "feature_name": v.feature_name,
                    "feature_type": v.feature_type,
                    "start_idx": v.start_idx,
                    "end_idx": v.end_idx,
                    "categories": v.categories,
                }
                for k, v in self.column_slices.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabularPreprocessor":
        """Reconstructs preprocessor from serialized dictionary."""
        instance = cls(
            numerical_features=data["numerical_features"],
            categorical_features=data["categorical_features"],
        )
        instance.num_means_ = {k: float(v) for k, v in data["num_means"].items()}
        instance.num_stds_ = {k: float(v) for k, v in data["num_stds"].items()}
        instance.num_medians_ = {k: float(v) for k, v in data["num_medians"].items()}
        instance.cat_categories_ = {k: list(v) for k, v in data["cat_categories"].items()}
        instance.cat_modes_ = {k: str(v) for k, v in data["cat_modes"].items()}
        instance.total_encoded_dim = int(data["total_encoded_dim"])

        instance.column_slices = {}
        for k, v in data["column_slices"].items():
            instance.column_slices[k] = ColumnSlice(
                feature_name=v["feature_name"],
                feature_type=v["feature_type"],
                start_idx=int(v["start_idx"]),
                end_idx=int(v["end_idx"]),
                categories=v.get("categories"),
            )

        instance.is_fitted = True
        return instance
