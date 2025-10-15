"""Tests for feature engineering and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_preprocessor,
    build_pipeline,
    simple_leakage_checks,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)
from sklearn.linear_model import LogisticRegression


def create_sample_data(n_rows: int = 100) -> pd.DataFrame:
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        "tenure_days": np.random.randint(1, 1000, n_rows),
        "tickets_last_30d": np.random.randint(0, 10, n_rows),
        "avg_handle_time": np.random.uniform(300, 900, n_rows),
        "sentiment_avg": np.random.uniform(-2, 2, n_rows),
        "escalations_90d": np.random.randint(0, 5, n_rows),
        "channel": np.random.choice(["phone", "email", "chat"], n_rows),
        "plan_tier": np.random.choice(["basic", "plus", "pro"], n_rows),
        "first_contact_resolution": np.random.choice([0, 1], n_rows),
        "churned": np.random.choice([0, 1], n_rows)
    })
    
    return df


def test_build_preprocessor():
    """Test that preprocessor can be built and fitted."""
    preprocessor = build_preprocessor()
    
    # Check that it has the right structure
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2  # numerical and categorical
    
    # Fit on sample data
    df = create_sample_data(50)
    X = df.drop(columns=["churned"])
    
    X_transformed = preprocessor.fit_transform(X)
    
    # Check output shape
    assert X_transformed.shape[0] == 50
    # Should have 5 numerical + encoded categorical features
    # channel (3) + plan_tier (3) + first_contact_resolution (2) = 8 categorical
    # Total: 5 + 8 = 13
    assert X_transformed.shape[1] == 13


def test_build_pipeline():
    """Test that full pipeline can be built and fitted."""
    estimator = LogisticRegression(max_iter=100, random_state=42)
    pipeline = build_pipeline(estimator)
    
    # Check pipeline structure
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "classifier"
    
    # Fit on sample data
    df = create_sample_data(50)
    X = df.drop(columns=["churned"])
    y = df["churned"]
    
    pipeline.fit(X, y)
    
    # Test prediction
    y_pred = pipeline.predict(X)
    assert len(y_pred) == 50
    assert all(pred in [0, 1] for pred in y_pred)
    
    # Test probability prediction
    y_prob = pipeline.predict_proba(X)
    assert y_prob.shape == (50, 2)
    assert np.allclose(y_prob.sum(axis=1), 1.0)


def test_simple_leakage_checks_pass():
    """Test that valid data passes leakage checks."""
    df = create_sample_data(50)
    
    # Should not raise any exceptions
    simple_leakage_checks(df)


def test_simple_leakage_checks_fail_negative():
    """Test that negative values in non-negative columns are caught."""
    df = create_sample_data(50)
    df.loc[0, "tenure_days"] = -10  # Invalid negative value
    
    with pytest.raises(ValueError, match="Negative values"):
        simple_leakage_checks(df)


def test_simple_leakage_checks_invalid_channel():
    """Test that invalid channel values are caught."""
    df = create_sample_data(50)
    df.loc[0, "channel"] = "invalid_channel"
    
    with pytest.raises(ValueError, match="Invalid channel"):
        simple_leakage_checks(df)


def test_simple_leakage_checks_invalid_plan_tier():
    """Test that invalid plan_tier values are caught."""
    df = create_sample_data(50)
    df.loc[0, "plan_tier"] = "platinum"
    
    with pytest.raises(ValueError, match="Invalid plan_tier"):
        simple_leakage_checks(df)


def test_preprocessor_handles_unknown_categories():
    """Test that preprocessor handles unknown categorical values gracefully."""
    preprocessor = build_preprocessor()
    
    # Train on subset of categories
    df_train = pd.DataFrame({
        "tenure_days": [100, 200],
        "tickets_last_30d": [1, 2],
        "avg_handle_time": [500, 600],
        "sentiment_avg": [0.5, -0.5],
        "escalations_90d": [0, 1],
        "channel": ["phone", "email"],
        "plan_tier": ["basic", "plus"],
        "first_contact_resolution": [0, 1]
    })
    
    preprocessor.fit(df_train)
    
    # Test with unseen category
    df_test = pd.DataFrame({
        "tenure_days": [150],
        "tickets_last_30d": [3],
        "avg_handle_time": [550],
        "sentiment_avg": [0.0],
        "escalations_90d": [0],
        "channel": ["chat"],  # Unseen category
        "plan_tier": ["pro"],  # Unseen category
        "first_contact_resolution": [1]
    })
    
    # Should not raise exception due to handle_unknown='ignore'
    X_test_transformed = preprocessor.transform(df_test)
    assert X_test_transformed.shape[0] == 1


def test_feature_column_definitions():
    """Test that feature column lists are correctly defined."""
    assert len(NUMERICAL_COLS) == 5
    assert len(CATEGORICAL_COLS) == 3
    
    assert "tenure_days" in NUMERICAL_COLS
    assert "channel" in CATEGORICAL_COLS
    assert "plan_tier" in CATEGORICAL_COLS

