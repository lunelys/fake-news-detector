import os
import re
from typing import Optional, List

import joblib
import numpy as np
from unidecode import unidecode
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from textblob import TextBlob

try:
    from textblob import Blobber
    from textblob_fr import PatternTagger, PatternAnalyzer
    _TB_FR = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    _TB_FR_AVAILABLE = True
except Exception:
    _TB_FR_AVAILABLE = False
    _TB_FR = None


REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])


class PredictRequest(BaseModel):
    text: str
    lang: Optional[str] = None


class PredictResponse(BaseModel):
    predicted_label: str
    credibility_score: float
    probabilities: List[float]
    top_terms: List[str]
    sentiment: float
    emotion_scores: dict
    dominant_emotion: str
    explanation_text: str
    alert: bool


app = FastAPI(title="Bluesky Fake News API", version="1.0.0")


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _model_dir() -> str:
    return os.getenv(
        "MODEL_DIR",
        os.path.join(_project_root(), "bluesky-pipeline", "data", "06_models"),
    )


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_models():
    model_dir = _model_dir()
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    classifier = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    calibrated_path = os.path.join(model_dir, "calibrated_classifier.joblib")
    calibrated = joblib.load(calibrated_path) if os.path.exists(calibrated_path) else None
    return vectorizer, classifier, label_encoder, calibrated


def _explain_text(vectorizer, classifier, label_encoder, clean: str, top_n: int = 10) -> List[str]:
    X = vectorizer.transform([clean])
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_
    class_names = list(label_encoder.classes_)

    if coefs.shape[0] == 1 and len(class_names) == 2:
        coefs = np.vstack([-coefs[0], coefs[0]])

    pred_index = int(classifier.predict(X)[0])
    class_coefs = coefs[pred_index]
    contributions = class_coefs * X.toarray()[0]
    top_indices = np.argsort(contributions)[::-1][:top_n]
    return [feature_names[i] for i in top_indices if contributions[i] > 0]


def _sentiment(text: str, lang: Optional[str]) -> float:
    if lang and lang.startswith("fr") and _TB_FR_AVAILABLE:
        blob = _TB_FR(text)
        try:
            return float(blob.sentiment[0])
        except Exception:
            return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity


def _emotion_scores(text: str) -> tuple[dict, str]:
    try:
        from nrclex import NRCLex
        scores = NRCLex(text).affect_frequencies or {}
        dominant = max(scores, key=scores.get) if scores else "unknown"
        return scores, dominant
    except Exception:
        return {}, "unknown"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    REQUESTS.labels(endpoint="/predict").inc()
    with LATENCY.labels(endpoint="/predict").time():
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is empty.")

        try:
            vectorizer, classifier, label_encoder, calibrated = _load_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

        clean = clean_text(text)
        X = vectorizer.transform([clean])
        proba_model = calibrated or classifier
        proba = proba_model.predict_proba(X)[0]
        pred_index = int(np.argmax(proba))
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        credibility_score = float(np.max(proba))
        top_terms = _explain_text(vectorizer, classifier, label_encoder, clean)
        sentiment = _sentiment(clean, payload.lang)
        emotion_scores, dominant_emotion = _emotion_scores(clean)
        alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.7"))
        alert = credibility_score < alert_threshold
        explanation_text = (
            f"This post is classified as '{pred_label}' with a credibility score of "
            f"{credibility_score:.2f}. Dominant emotion: {dominant_emotion}. "
            f"Key terms: {', '.join(top_terms[:5]) if top_terms else 'none'}."
        )

        return PredictResponse(
            predicted_label=pred_label,
            credibility_score=credibility_score,
            probabilities=[float(p) for p in proba],
            top_terms=top_terms,
            sentiment=sentiment,
            emotion_scores=emotion_scores,
            dominant_emotion=dominant_emotion,
            explanation_text=explanation_text,
            alert=alert,
        )
