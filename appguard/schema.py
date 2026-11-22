"""Schema definitions for AppGuard AI insurance applications.

Adheres strictly to privacy and operational guidelines:
- Excludes names, government identification numbers (SSN, TIN, passport),
  exact dates of birth, street addresses, protected demographic traits
  (race, religion, sex, sexual orientation, disability status), and obvious proxies.
- Includes operational, transactional, and policy-structure variables.
- Educational demonstration schema using synthetic data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


# Categorical valid options
POLICY_TYPES = ["auto", "homeowners", "term_life", "renters", "commercial_property"]
PREMIUM_FREQUENCIES = ["monthly", "quarterly", "semi_annual", "annual"]
APPLICATION_CHANNELS = ["online_direct", "independent_broker", "captive_agent", "aggregator_partner", "mobile_app"]
BENEFICIARY_RELATIONS = ["self", "spouse", "child", "parent", "business_partner", "other"]
DECLARED_CREDIT_TIERS = ["excellent", "good", "fair", "limited_history"]

NUMERICAL_FEATURES: List[str] = [
    "coverage_amount",
    "deductible_amount",
    "applicant_age",
    "annual_income_estimate",
    "prior_application_count",
    "recent_address_changes_count",
    "property_or_vehicle_age",
]

CATEGORICAL_FEATURES: List[str] = [
    "policy_type",
    "premium_frequency",
    "application_channel",
    "beneficiary_relation",
    "declared_credit_tier",
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class InsuranceApplicationInput(BaseModel):
    """Pydantic model representing an individual incoming policy application for scoring."""
    model_config = ConfigDict(extra="ignore")

    application_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for tracking/audit. Not used as a model feature."
    )
    policy_type: str = Field(
        ...,
        description=f"Type of insurance policy requested. Allowed: {POLICY_TYPES}"
    )
    coverage_amount: float = Field(
        ...,
        gt=0,
        le=10_000_000,
        description="Requested coverage limit in USD."
    )
    deductible_amount: float = Field(
        ...,
        ge=0,
        le=100_000,
        description="Policy deductible in USD."
    )
    premium_frequency: str = Field(
        ...,
        description=f"Payment frequency schedule. Allowed: {PREMIUM_FREQUENCIES}"
    )
    application_channel: str = Field(
        ...,
        description=f"Origination channel of the submission. Allowed: {APPLICATION_CHANNELS}"
    )
    applicant_age: float = Field(
        ...,
        ge=18,
        le=105,
        description="Declared age of primary applicant (minimum 18)."
    )
    annual_income_estimate: float = Field(
        ...,
        ge=0,
        le=5_000_000,
        description="Estimated annual gross income in USD."
    )
    prior_application_count: float = Field(
        default=0.0,
        ge=0,
        le=50,
        description="Prior policy applications submitted across insurers within the past 3 years."
    )
    recent_address_changes_count: float = Field(
        default=0.0,
        ge=0,
        le=20,
        description="Number of residential address transitions recorded in the past 24 months."
    )
    property_or_vehicle_age: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Age in years of the primary insured vehicle, structure, or equipment."
    )
    beneficiary_relation: str = Field(
        default="spouse",
        description=f"Relationship of designated primary beneficiary. Allowed: {BENEFICIARY_RELATIONS}"
    )
    declared_credit_tier: str = Field(
        default="good",
        description=f"Self-declared creditworthiness tier. Allowed: {DECLARED_CREDIT_TIERS}"
    )

    @field_validator("policy_type")
    @classmethod
    def validate_policy_type(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in POLICY_TYPES:
            raise ValueError(f"Invalid policy_type '{v}'. Must be one of {POLICY_TYPES}")
        return clean

    @field_validator("premium_frequency")
    @classmethod
    def validate_premium_frequency(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in PREMIUM_FREQUENCIES:
            raise ValueError(f"Invalid premium_frequency '{v}'. Must be one of {PREMIUM_FREQUENCIES}")
        return clean

    @field_validator("application_channel")
    @classmethod
    def validate_application_channel(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in APPLICATION_CHANNELS:
            raise ValueError(f"Invalid application_channel '{v}'. Must be one of {APPLICATION_CHANNELS}")
        return clean

    @field_validator("beneficiary_relation")
    @classmethod
    def validate_beneficiary_relation(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in BENEFICIARY_RELATIONS:
            raise ValueError(f"Invalid beneficiary_relation '{v}'. Must be one of {BENEFICIARY_RELATIONS}")
        return clean

    @field_validator("declared_credit_tier")
    @classmethod
    def validate_declared_credit_tier(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in DECLARED_CREDIT_TIERS:
            raise ValueError(f"Invalid declared_credit_tier '{v}'. Must be one of {DECLARED_CREDIT_TIERS}")
        return clean


@dataclass
class FeatureDeviation:
    """Represents a feature whose reconstruction differs significantly from typical applications."""
    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    submitted_value: Any
    reconstructed_value: Any
    reconstruction_error: float
    deviation_percentile: float
    statistical_observation: str


@dataclass
class ApplicationScoreResult:
    """Complete scoring result for an insurance application."""
    application_id: Optional[str]
    anomaly_score: float
    calibrated_threshold: float
    review_recommended: bool
    review_priority: str  # "standard", "elevated", "high"
    top_deviations: List[FeatureDeviation]
    neutral_summary: str
    regulatory_notice: str
