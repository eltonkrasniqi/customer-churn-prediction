import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import RANDOM_STATE, TEST_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def train_test_split_time_safe(
    df: pd.DataFrame,
    target: str = "churned",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    logger.info(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test
