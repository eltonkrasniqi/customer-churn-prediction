import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from sklearn.linear_model import LogisticRegression

from src.core.config import MAX_ITERATIONS, RANDOM_STATE, RISK_THRESHOLD_LOW, RISK_THRESHOLD_MEDIUM, TEST_SIZE
from src.data.dataset import train_test_split_time_safe
from src.features import build_pipeline
from src.schema import ChurnPredictionRequest, ChurnPredictionResponse, ReadyResponse

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

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json"
)


def load_model(path: Path) -> any:
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def train_and_save_model() -> any:
    logger.info("Training new model from sample data")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. "
            "Please ensure data/raw/churn_sample.csv exists."
        )
    
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split_time_safe(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    estimator = LogisticRegression(max_iter=MAX_ITERATIONS, class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = build_pipeline(estimator)
    pipeline.fit(X_train, y_train)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return pipeline


def get_model() -> any:
    if not hasattr(app.state, "model"):
        app.state.model = None
    
    if app.state.model is not None:
        return app.state.model
    
    if MODEL_PATH.exists():
        app.state.model = load_model(MODEL_PATH)
        return app.state.model
    
    logger.warning(f"Model not found at {MODEL_PATH}, training new model")
    app.state.model = train_and_save_model()
    return app.state.model


@app.on_event("startup")
async def startup_event():
    app.state.model = None
    if MODEL_PATH.exists():
        try:
            app.state.model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load model on startup: {e}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Will train on first prediction request.")


@app.get("/")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    
    if "text/html" in accept or not accept:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    else:
        return {
            "message": "Churn Prediction API",
            "version": "0.1.0",
            "docs": "/docs"
        }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
async def ready_check():
    return ReadyResponse(ready=hasattr(app.state, "model") and app.state.model is not None)


@app.get("/api/analytics")
async def get_analytics():
    try:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = pd.read_csv(DATA_PATH)
        
        model = get_model()
        
        feature_cols = [
            "tenure_days", "tickets_last_30d", "avg_handle_time",
            "first_contact_resolution", "sentiment_avg", "escalations_90d",
            "channel", "plan_tier"
        ]
        X = df[feature_cols].copy()
        
        predictions = model.predict_proba(X)[:, 1]
        df['churn_probability'] = predictions
        
        df['risk_band'] = df['churn_probability'].apply(
            lambda p: "Low" if p < RISK_THRESHOLD_LOW else "Medium" if p < RISK_THRESHOLD_MEDIUM else "High"
        )
        
        analytics = {
            "total_customers": len(df),
            "churn_rate": float(df['churned'].mean()) if 'churned' in df.columns else 0.0,
            "high_risk_count": int((df['risk_band'] == 'High').sum()),
            "risk_bands": {
                "low": int((df['risk_band'] == 'Low').sum()),
                "medium": int((df['risk_band'] == 'Medium').sum()),
                "high": int((df['risk_band'] == 'High').sum())
            },
            "probability_bins": {
                "labels": ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                          '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                "counts": [int(x) for x in pd.cut(df['churn_probability'], bins=10).value_counts().sort_index().values]
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    try:
        model = get_model()
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model unavailable",
                "message": str(e),
                "instructions": "Run 'python -m src.train' or 'make train' to create the model"
            }
        )
    
    try:
        data = {
            "tenure_days": [request.tenure_days],
            "tickets_last_30d": [request.tickets_last_30d],
            "avg_handle_time": [request.avg_handle_time],
            "first_contact_resolution": [request.first_contact_resolution],
            "sentiment_avg": [request.sentiment_avg],
            "escalations_90d": [request.escalations_90d],
            "channel": [request.channel.value],
            "plan_tier": [request.plan_tier.value]
        }
        df = pd.DataFrame(data)
        
        probability = float(model.predict_proba(df)[0, 1])
        
        if probability < RISK_THRESHOLD_LOW:
            risk_band = "Low"
        elif probability < RISK_THRESHOLD_MEDIUM:
            risk_band = "Medium"
        else:
            risk_band = "High"
        
        return ChurnPredictionResponse(
            churn_probability=round(probability, 4),
            risk_band=risk_band,
            model_version="latest"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
