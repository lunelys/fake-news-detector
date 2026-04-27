# FastAPI service

Run:
1. `pip install -r requirements_app.txt`
2. `uvicorn apps.api.main:app --reload --port 8000`

Endpoints:
- `GET /health`
- `POST /predict` with JSON: `{ "text": "your text", "lang": "en" }`
- `GET /metrics` for Prometheus
