import logging
import sys
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

NUMERICAL_COLS = [
    "tenure_days",
    "tickets_last_30d",
    "avg_handle_time",
    "sentiment_avg",
    "escalations_90d"
]

CATEGORICAL_COLS = [
    "channel",
    "plan_tier",
    "first_contact_resolution"
]


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        drop=None
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def simple_leakage_checks(df: pd.DataFrame) -> None:
    logger.info("Running data quality checks")
    
    issues = []
    
    non_negative_cols = [
        "tenure_days", "tickets_last_30d", "avg_handle_time", "escalations_90d"
    ]
    for col in non_negative_cols:
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"Negative values found in {col}")
    
    required_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    for col in required_cols:
        if col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct > 0.5:
                issues.append(f"High missing rate in {col}: {missing_pct:.1%}")
    
    if "channel" in df.columns:
        valid_channels = {"phone", "email", "chat"}
        invalid = set(df["channel"].unique()) - valid_channels
        if invalid:
            issues.append(f"Invalid channel values: {invalid}")
    
    if "plan_tier" in df.columns:
        valid_tiers = {"basic", "plus", "pro"}
        invalid = set(df["plan_tier"].unique()) - valid_tiers
        if invalid:
            issues.append(f"Invalid plan_tier values: {invalid}")
    
    if "first_contact_resolution" in df.columns:
        valid_fcr = {0, 1}
        invalid = set(df["first_contact_resolution"].unique()) - valid_fcr
        if invalid:
            issues.append(f"Invalid first_contact_resolution values: {invalid}")
    
    if issues:
        error_msg = "Data quality issues:\n  - " + "\n  - ".join(issues)
        raise ValueError(error_msg)
    
    logger.info("Data quality checks passed")


def build_pipeline(estimator, preprocessor: Optional[ColumnTransformer] = None) -> Pipeline:
    if preprocessor is None:
        preprocessor = build_preprocessor()
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    
    return pipeline
