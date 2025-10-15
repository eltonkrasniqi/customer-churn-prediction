import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from src.core.config import MAX_ITERATIONS, RANDOM_STATE, TEST_SIZE
from src.data.dataset import train_test_split_time_safe
from src.features import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_PATH = MODELS_DIR / "model.joblib"
DATA_PATH = DATA_DIR / "churn_sample.csv"


def main():
    logger.info("Starting training")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows")
    
    X_train, X_test, y_train, y_test = train_test_split_time_safe(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info("Training Logistic Regression model")
    estimator = LogisticRegression(max_iter=MAX_ITERATIONS, class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = build_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"Accuracy: {accuracy:.4f}, ROC AUC: {auc:.4f}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
