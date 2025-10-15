import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from src.data.dataset import train_test_split_time_safe

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
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.info("Run training first: python -m src.train")
        return
    
    if not DATA_PATH.exists():
        logger.error(f"Data not found: {DATA_PATH}")
        return
    
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split_time_safe(
        df, test_size=0.2, random_state=42
    )
    
    logger.info(f"Evaluating on {len(X_test)} test samples")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {auc:.4f}")
    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned'])}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
