import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_ready_endpoint_initial(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert isinstance(data["ready"], bool)


def test_predict_endpoint_triggers_training_if_needed(client):
    payload = {
        "tenure_days": 120,
        "tickets_last_30d": 5,
        "avg_handle_time": 650.0,
        "first_contact_resolution": 0,
        "sentiment_avg": -1.2,
        "escalations_90d": 2,
        "channel": "email",
        "plan_tier": "basic"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "churn_probability" in data
    assert "risk_band" in data
    assert "model_version" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_band"] in ["Low", "Medium", "High"]


def test_ready_endpoint_after_prediction(client):
    payload = {
        "tenure_days": 120,
        "tickets_last_30d": 5,
        "avg_handle_time": 650.0,
        "first_contact_resolution": 0,
        "sentiment_avg": -1.2,
        "escalations_90d": 2,
        "channel": "email",
        "plan_tier": "basic"
    }
    
    client.post("/predict", json=payload)
    
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True


def test_predict_invalid_channel(client):
    payload = {
        "tenure_days": 120,
        "tickets_last_30d": 5,
        "avg_handle_time": 650.0,
        "first_contact_resolution": 0,
        "sentiment_avg": -1.2,
        "escalations_90d": 2,
        "channel": "invalid_channel",
        "plan_tier": "basic"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field(client):
    payload = {
        "tenure_days": 120,
        "tickets_last_30d": 5,
        "first_contact_resolution": 0,
        "sentiment_avg": -1.2,
        "escalations_90d": 2,
        "channel": "email",
        "plan_tier": "basic"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_multiple_requests(client):
    high_risk = {
        "tenure_days": 30,
        "tickets_last_30d": 10,
        "avg_handle_time": 800.0,
        "first_contact_resolution": 0,
        "sentiment_avg": -2.0,
        "escalations_90d": 5,
        "channel": "phone",
        "plan_tier": "basic"
    }
    
    low_risk = {
        "tenure_days": 1000,
        "tickets_last_30d": 1,
        "avg_handle_time": 400.0,
        "first_contact_resolution": 1,
        "sentiment_avg": 2.0,
        "escalations_90d": 0,
        "channel": "chat",
        "plan_tier": "pro"
    }
    
    resp1 = client.post("/predict", json=high_risk)
    resp2 = client.post("/predict", json=low_risk)
    
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    
    prob1 = resp1.json()["churn_probability"]
    prob2 = resp2.json()["churn_probability"]
    
    assert prob1 > prob2
