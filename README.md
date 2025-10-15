# Churn Prediction Service

Customer churn prediction system with ML training pipeline and REST API.

**Features:**
- Predict customer churn probability with risk banding
- REST API for real-time predictions

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make train
make serve
```

Visit `http://127.0.0.1:8000`

---

## Dataset

Uses `data/raw/churn_sample.csv` committed in the repo.

---

## Commands

```bash
make install    # Install dependencies
make train      # Train the model
make eval       # Evaluate the model
make serve      # Start API server
make test       # Run tests
make lint       # Run linter
make format     # Format code
```

---

## Training & Evaluation

```bash
make train
make eval
```

**Outputs:** `models/model.joblib`

---

## API

```bash
make serve
# or: uvicorn src.api.app:app --reload --port 8000
```

**Endpoints:**
- `GET /health` - Health check
- `GET /ready` - Readiness (model loaded)
- `POST /predict` - Predict churn probability and risk band

**Risk Bands:** Low (<0.3), Medium (0.3-0.6), High (≥0.6)

---

## Testing

```bash
make test
```

---

## Configuration

No external configuration required for basic usage.

---

## License

[MIT License](LICENSE)