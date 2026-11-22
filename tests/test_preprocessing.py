"""Unit tests for TabularPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from appguard.preprocessor import TabularPreprocessor
from appguard.schema import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "coverage_amount": [50000.0, 100000.0, 250000.0, 500000.0],
        "deductible_amount": [500.0, 1000.0, 1500.0, 2000.0],
        "applicant_age": [25.0, 35.0, 45.0, 55.0],
        "annual_income_estimate": [40000.0, 75000.0, 120000.0, 200000.0],
        "prior_application_count": [0.0, 1.0, 0.0, 2.0],
        "recent_address_changes_count": [0.0, 1.0, 0.0, 0.0],
        "property_or_vehicle_age": [2.0, 5.0, 8.0, 12.0],
        "policy_type": ["auto", "homeowners", "term_life", "auto"],
        "premium_frequency": ["monthly", "annual", "monthly", "quarterly"],
        "application_channel": ["online_direct", "captive_agent", "independent_broker", "mobile_app"],
        "beneficiary_relation": ["spouse", "child", "spouse", "self"],
        "declared_credit_tier": ["good", "excellent", "fair", "good"],
    })


def test_preprocessor_fit_transform(sample_dataframe):
    prep = TabularPreprocessor()
    prep.fit(sample_dataframe)

    assert prep.is_fitted
    assert prep.total_encoded_dim > len(NUMERICAL_FEATURES)

    encoded = prep.transform(sample_dataframe)
    assert encoded.shape == (len(sample_dataframe), prep.total_encoded_dim)
    assert not np.isnan(encoded).any()
    assert not np.isinf(encoded).any()


def test_preprocessor_handles_missing_values(sample_dataframe):
    prep = TabularPreprocessor()
    prep.fit(sample_dataframe)

    # DataFrame with missing numerical and categorical values
    dirty_df = pd.DataFrame({
        "coverage_amount": [None, 100000.0],
        "applicant_age": [np.nan, 40.0],
        "policy_type": [None, "auto"],
        "application_channel": ["unknown_channel", None],
    })

    encoded = prep.transform(dirty_df)
    assert encoded.shape == (2, prep.total_encoded_dim)
    assert not np.isnan(encoded).any()
    assert not np.isinf(encoded).any()


def test_preprocessor_handles_unknown_categories(sample_dataframe):
    prep = TabularPreprocessor()
    prep.fit(sample_dataframe)

    unseen_df = pd.DataFrame([{
        "coverage_amount": 50000.0,
        "deductible_amount": 500.0,
        "applicant_age": 30.0,
        "annual_income_estimate": 60000.0,
        "prior_application_count": 0.0,
        "recent_address_changes_count": 0.0,
        "property_or_vehicle_age": 3.0,
        "policy_type": "alien_policy_type_999",  # unknown category
        "premium_frequency": "weekly_not_seen",  # unknown category
        "application_channel": "telepathy",      # unknown category
        "beneficiary_relation": "pet",           # unknown category
        "declared_credit_tier": "unheard_tier",  # unknown category
    }])

    encoded = prep.transform(unseen_df)
    assert encoded.shape == (1, prep.total_encoded_dim)
    # Categorical slices should safely be all zeroes without raising an exception
    policy_slice = prep.column_slices["policy_type"]
    assert np.all(encoded[0, policy_slice.start_idx:policy_slice.end_idx] == 0.0)


def test_preprocessor_serialization(sample_dataframe):
    prep1 = TabularPreprocessor()
    prep1.fit(sample_dataframe)
    state = prep1.to_dict()

    prep2 = TabularPreprocessor.from_dict(state)
    assert prep2.is_fitted
    assert prep2.total_encoded_dim == prep1.total_encoded_dim

    t1 = prep1.transform(sample_dataframe)
    t2 = prep2.transform(sample_dataframe)
    np.testing.assert_allclose(t1, t2)
