# FastAPI service

Run:
1. `pip install -r requirements_app.txt`
2. `uvicorn apps.api.main:app --reload --port 8000`

Endpoints:
- `GET /health`
- `POST /predict` with JSON: `{ "text": "your text", "lang": "en" }`
- `GET /metrics` for Prometheus

`/predict` is used by the Streamlit dashboard for live post analysis. The response includes the predicted label, model confidence, class probabilities, key TF-IDF terms, sentiment, emotion scores, dominant emotion, user-facing explanation text, and alert flag.

Emotion scores come from the current fallback emotion detector. If no emotion terms are recognized, scores are zero and the dominant emotion is `unknown`.
