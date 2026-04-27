from pymongo import MongoClient, UpdateOne
from typing import List, Dict, Optional, Tuple
from unidecode import unidecode
import re
import spacy
import numpy as np
from langdetect import detect
from sqlalchemy import create_engine, text
from scipy.spatial.distance import jensenshannon

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

import joblib
import os
import json
import pandas as pd
import glob
from datetime import datetime, timezone
import warnings

try:
    from nrclex import NRCLex
    _NRCLEX_AVAILABLE = True
except Exception:
    _NRCLEX_AVAILABLE = False

try:
    from textblob import Blobber
    from textblob_fr import PatternTagger, PatternAnalyzer
    _TB_FR = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    _TB_FR_AVAILABLE = True
except Exception:
    _TB_FR_AVAILABLE = False
    _TB_FR = None


# ============================================================
# LOAD SPACY MODELS
# ============================================================

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


def detect_language_from_post(post: Dict, fallback: str = "en") -> str:
    langs = post.get("langs", [])
    if langs:
        return langs[0]

    text = post.get("clean_text", "")
    if not text:
        return fallback

    try:
        return detect(text)
    except Exception:
        return fallback


# ============================================================
# LOAD DATA FROM MONGODB
# ============================================================

def load_raw_posts(mongo_uri: str, db_name: str, raw_collections: List[str]) -> List[Dict]:
    """
    Load and merge posts from multiple MongoDB collections.

    Each collection corresponds to a supervised class label.
    The collection name is injected as `source_label`.

    Returns:
        List of enriched post dictionaries.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    all_posts = []

    for col_name in raw_collections:
        for post in db[col_name].find({}):
            post["source_label"] = col_name
            all_posts.append(post)

    return all_posts


def extract_text(post: dict) -> str:
    """
    Extract primary text content from a Bluesky post.
    Handles nested embed structures.
    """
    text = post.get("record", {}).get("text")
    if text:
        return text

    embed_record = post.get("record", {}).get("embed", {}).get("record", {}).get("record", {})
    return embed_record.get("text", "") or ""


# ============================================================
# CLEANING
# ============================================================

def clean_text_node(posts: List[Dict]) -> List[Dict]:
    """
    Normalize post text.

    Steps:
        - Lowercasing
        - Remove URLs, mentions, hashtags
        - Remove punctuation
        - Remove accents
        - Normalize whitespace

    Adds:
        clean_text field to each post
    """
    cleaned = []

    for post in posts:
        text = extract_text(post)
        if not text:
            continue

        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = unidecode(text)
        text = re.sub(r"\s+", " ", text)

        post["clean_text"] = text.strip()
        cleaned.append(post)

    return cleaned


# ============================================================
# TOKENIZATION + LEMMATIZATION
# ============================================================

def tokenize_and_lemmatize(posts: List[Dict]) -> List[Dict]:
    """
    Language-aware lemmatization using SpaCy.

    Strategy:
        - Use Bluesky 'langs' if available
        - Fallback to langdetect
        - Remove stopwords
        - Keep alphabetic tokens only

    Adds:
        tokens field
    """
    processed = []

    for post in posts:
        text = post.get("clean_text", "")
        if not text:
            continue

        lang = detect_language_from_post(post)

        if lang.startswith("fr"):
            doc = nlp_fr(text)
        else:
            doc = nlp_en(text)

        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha
        ]

        post["tokens"] = tokens
        processed.append(post)

    return processed


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(posts: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicates based on clean_text.
    """
    seen = set()
    unique_posts = []
    duplicates_count = 0

    for post in posts:
        text = post.get("clean_text", "")
        if text and text not in seen:
            seen.add(text)
            unique_posts.append(post)
        else:
            duplicates_count += 1

    print(f"[Deduplication] Final dataset size: {len(unique_posts)}, Duplicates removed: {duplicates_count}")
    return unique_posts


# ============================================================
# CREDIBILITY LABELS (WEAK SUPERVISION)
# ============================================================

def derive_credibility_labels(posts: List[Dict], verified_sources: List[str]) -> List[str]:
    """
    Create a weakly-supervised credibility label.

    Verified sources -> "credible"
    Others -> "unverified"
    """
    labels = []
    for post in posts:
        source = post.get("source_label", "")
        label = "credible" if source in verified_sources else "unverified"
        post["credibility_label"] = label
        labels.append(label)
    return labels


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

def vectorize_posts(
    posts: List[Dict],
    max_features: int = 5000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
    use_stopwords: bool = True,
    custom_stopwords: Optional[List[str]] = None,
):
    """
    Convert cleaned text into TF-IDF feature matrix.

    Uses:
        - 1-2 grams
        - min_df=2
        - feature cap

    Returns:
        X, labels, vectorizer
    """
    texts = [p["clean_text"] for p in posts]
    labels = [p["source_label"] for p in posts]

    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)

    stopwords = None
    if use_stopwords:
        stopwords = set()
        stopwords.update(nlp_en.Defaults.stop_words)
        stopwords.update(nlp_fr.Defaults.stop_words)
        if custom_stopwords:
            stopwords.update([w.strip().lower() for w in custom_stopwords if w.strip()])

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=list(stopwords) if stopwords else None,
    )

    X = vectorizer.fit_transform(texts)

    print(f"[Vectorization] Matrix shape: {X.shape}")
    return X, labels, vectorizer


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def add_sentiment(posts: List[Dict]) -> List[Dict]:
    """
    Compute sentiment polarity using TextBlob.

    Sentiment range:
        -1 (negative) to +1 (positive)
    """
    for post in posts:
        text = post.get("clean_text", "")
        if not text:
            post["sentiment"] = 0.0
            post["lang_detected"] = "unknown"
            continue

        lang = detect_language_from_post(post)
        post["lang_detected"] = lang

        if lang.startswith("fr") and _TB_FR_AVAILABLE:
            blob = _TB_FR(text)
            try:
                post["sentiment"] = float(blob.sentiment[0])
            except Exception:
                post["sentiment"] = 0.0
        else:
            blob = TextBlob(text)
            post["sentiment"] = blob.sentiment.polarity
    return posts


def compute_sentiment_summary(posts: List[Dict]):
    """
    Compute descriptive statistics of sentiment per label.
    Save summary as JSON.
    """
    os.makedirs("data/08_reporting", exist_ok=True)

    df = pd.DataFrame(posts)
    summary = df.groupby("source_label")["sentiment"].agg(["mean", "std", "count"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary.to_json(f"data/08_reporting/sentiment_summary_{timestamp}.json")

    print("[Sentiment] Summary saved.")


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def add_emotions(
    posts: List[Dict],
    use_transformer: bool = False,
    model_name: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict]:
    """
    Add basic emotion signals using NRC Lexicon (English-focused).

    Adds:
        - emotion_scores: dict of emotion -> score
        - dominant_emotion: highest-score emotion or "unknown"
    """
    emotion_failures = 0

    if use_transformer:
        try:
            from transformers import pipeline
            emotion_model = model_name or "cardiffnlp/twitter-xlm-roberta-base-emotion"
            emo_pipe = pipeline("text-classification", model=emotion_model, top_k=None)
            for post in posts:
                text = post.get("clean_text", "")
                if not text:
                    post["emotion_scores"] = {}
                    post["dominant_emotion"] = "unknown"
                    continue
                scores = emo_pipe(text)[0]
                score_map = {s["label"].lower(): float(s["score"]) for s in scores}
                dominant = max(score_map, key=score_map.get) if score_map else "unknown"
                post["emotion_scores"] = score_map
                post["dominant_emotion"] = dominant
                post["emotion_status"] = "ok"
            return posts
        except Exception:
            print("[Emotion] Transformer model unavailable, falling back to NRC Lexicon.")

    for post in posts:
        text = post.get("clean_text", "")
        if not text:
            post["emotion_scores"] = {}
            post["dominant_emotion"] = "unknown"
            post["emotion_status"] = "missing_text"
            continue

        if not _NRCLEX_AVAILABLE:
            post["emotion_scores"] = {}
            post["dominant_emotion"] = "unknown"
            post["emotion_status"] = "nrcl_ex_unavailable"
            emotion_failures += 1
            continue

        try:
            nrc = NRCLex(text)
            scores = nrc.affect_frequencies or {}
        except Exception as exc:
            print(f"[Emotion] NRCLex/TextBlob resources unavailable ({exc}). Falling back to empty emotion scores.")
            post["emotion_scores"] = {}
            post["dominant_emotion"] = "unknown"
            post["emotion_status"] = "emotion_failed"
            emotion_failures += 1
            continue

        if scores:
            dominant = max(scores, key=scores.get)
        else:
            dominant = "unknown"

        post["emotion_scores"] = scores
        post["dominant_emotion"] = dominant
        post["emotion_status"] = "ok"

    if emotion_failures > 0:
        warning_message = (
            f"Emotion analysis failed for {emotion_failures} post(s). "
            "Results were filled with empty emotion scores/unknown dominant emotion. "
            "Install the required corpora and rerun the pipeline for complete emotion outputs."
        )
        warnings.warn(warning_message)
        print(f"[Emotion][WARNING] {warning_message}")

    return posts


def build_user_facing_explanations(posts: List[Dict]) -> List[Dict]:
    """
    Build a readable explanation string for dashboards and API consumers.
    """
    for post in posts:
        predicted_label = post.get("predicted_label", post.get("credibility_label", "unknown"))
        credibility_score = float(post.get("credibility_score", 0.0))
        dominant_emotion = post.get("dominant_emotion", "unknown")
        explanation_terms = post.get("explanation_terms", [])

        if explanation_terms:
            terms_text = ", ".join(explanation_terms[:5])
            reason = f"Key terms influencing the score: {terms_text}."
        else:
            reason = "No strong keyword explanation was available."

        post["explanation_text"] = (
            f"This post is classified as '{predicted_label}' "
            f"with a credibility score of {credibility_score:.2f}. "
            f"Dominant emotion detected: {dominant_emotion}. {reason}"
        )

    return posts


def compute_emotion_summary(posts: List[Dict]):
    """
    Compute distribution of dominant emotions per label.
    Save summary as JSON.
    """
    os.makedirs("data/08_reporting", exist_ok=True)

    df = pd.DataFrame(posts)
    if "dominant_emotion" not in df.columns:
        print("[Emotion] No dominant_emotion column found. Skipping summary.")
        return

    summary = (
        df.groupby(["source_label", "dominant_emotion"])
        .size()
        .unstack(fill_value=0)
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary.to_json(f"data/08_reporting/emotion_summary_{timestamp}.json")
    print("[Emotion] Summary saved.")


# ============================================================
# DATA DRIFT REPORTING
# ============================================================

def _load_latest_distribution(pattern: str) -> Optional[Dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    latest = files[-1]
    try:
        with open(latest, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _js_divergence(current: Dict, previous: Dict) -> Optional[float]:
    keys = sorted(set(current.keys()) | set(previous.keys()))
    if not keys:
        return None

    p = np.array([current.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([previous.get(k, 0.0) for k in keys], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return None

    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q))


def compute_data_drift_report(posts: List[Dict]):
    """
    Compute simple distribution drift for labels and languages.
    Saves current distributions and Jensen-Shannon divergence vs the previous run.
    """
    os.makedirs("data/08_reporting", exist_ok=True)

    df = pd.DataFrame(posts)
    if df.empty:
        print("[Drift] No data available. Skipping.")
        return

    if "lang_detected" not in df.columns:
        df["lang_detected"] = df.apply(lambda r: detect_language_from_post(r.to_dict()), axis=1)

    label_dist = df["source_label"].value_counts(normalize=True).to_dict()
    lang_dist = df["lang_detected"].value_counts(normalize=True).to_dict()

    prev_labels = _load_latest_distribution("data/08_reporting/label_distribution_*.json")
    prev_langs = _load_latest_distribution("data/08_reporting/language_distribution_*.json")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    label_path = f"data/08_reporting/label_distribution_{timestamp}.json"
    lang_path = f"data/08_reporting/language_distribution_{timestamp}.json"

    with open(label_path, "w") as f:
        json.dump(label_dist, f, indent=4)
    with open(lang_path, "w") as f:
        json.dump(lang_dist, f, indent=4)

    drift_report = {
        "label_js_divergence": _js_divergence(label_dist, prev_labels) if prev_labels else None,
        "language_js_divergence": _js_divergence(lang_dist, prev_langs) if prev_langs else None,
        "current_label_distribution": label_dist,
        "current_language_distribution": lang_dist,
    }

    with open(f"data/08_reporting/data_drift_{timestamp}.json", "w") as f:
        json.dump(drift_report, f, indent=4)

    print("[Drift] Report saved.")


# ============================================================
# STORE TO POSTGRES
# ============================================================

def store_vectors_to_postgres(posts, X, postgres_uri, batch_size=1000):
    """
    Store TF-IDF embeddings in PostgreSQL using pgvector.
    Idempotent via ON CONFLICT on clean_text.
    """
    engine = create_engine(postgres_uri)
    vector_dim = X.shape[1]

    create_extension_stmt = text("CREATE EXTENSION IF NOT EXISTS vector")
    create_table_stmt = text(f"""
        CREATE TABLE IF NOT EXISTS posts_vectors (
            id BIGSERIAL PRIMARY KEY,
            clean_text TEXT UNIQUE NOT NULL,
            embedding VECTOR({vector_dim}) NOT NULL,
            label TEXT,
            sentiment DOUBLE PRECISION
        )
    """)

    stmt = text("""
        INSERT INTO posts_vectors (clean_text, embedding, label, sentiment)
        VALUES (:text, CAST(:vector AS vector), :label, :sentiment)
        ON CONFLICT (clean_text) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(create_extension_stmt)
        conn.execute(create_table_stmt)

        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            batch_posts = posts[start:end]
            batch_vectors = X[start:end]

            data = []
            for i, post in enumerate(batch_posts):
                vec_list = batch_vectors[i].toarray()[0].tolist()
                vec_literal = "[" + ",".join(str(v) for v in vec_list) + "]"
                data.append({
                    "text": post["clean_text"],
                    "vector": vec_literal,
                    "label": post["source_label"],
                    "sentiment": post.get("sentiment", 0.0)
                })

            conn.execute(stmt, data)

    print("[Postgres] Storage complete.")


# ============================================================
# STORE CLEANED POSTS TO MONGODB
# ============================================================

def store_cleaned_posts_to_mongo(posts: List[Dict], mongo_uri: str, db_name: str, collection_name: str):
    """
    Store cleaned/enriched posts back to MongoDB.
    Uses clean_text as a natural key to avoid duplicates.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    collection.create_index("clean_text", unique=True)

    if not posts:
        print("[Mongo] No posts to store.")
        return

    docs = []
    for post in posts:
        doc = {
            "uri": post.get("uri"),
            "source_label": post.get("source_label"),
            "credibility_label": post.get("credibility_label", "unverified"),
            "clean_text": post.get("clean_text"),
            "tokens": post.get("tokens", []),
            "lang_detected": post.get("lang_detected", "unknown"),
            "sentiment": post.get("sentiment", 0.0),
            "emotion_scores": post.get("emotion_scores", {}),
            "dominant_emotion": post.get("dominant_emotion", "unknown"),
            "explanation_terms": post.get("explanation_terms", []),
            "explanation_text": post.get("explanation_text"),
            "credibility_score": post.get("credibility_score"),
            "predicted_label": post.get("predicted_label"),
            "alert": post.get("alert"),
            "inserted_at": datetime.now(timezone.utc),
        }
        docs.append(doc)

    inserted = 0
    for doc in docs:
        try:
            collection.insert_one(doc)
            inserted += 1
        except Exception:
            continue

    print(f"[Mongo] Stored {inserted} cleaned posts in '{collection_name}'.")


def store_explanations_to_mongo(
    posts: List[Dict],
    X,
    classifier,
    label_encoder,
    vectorizer,
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    top_n: int = 10,
    calibrated_classifier=None,
    alert_threshold: float = 0.7,
):
    """
    Store per-post explanations (top contributing terms) alongside MongoDB documents.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_

    if coefs.shape[0] == 1:
        coefs = np.vstack([-coefs[0], coefs[0]])

    requests = []
    proba_model = calibrated_classifier or classifier

    for idx, post in enumerate(posts):
        clean_text = post.get("clean_text")
        if not clean_text:
            continue

        vec = X[idx].toarray()[0]
        pred_index = int(classifier.predict(X[idx])[0])
        contrib = coefs[pred_index] * vec
        top_indices = np.argsort(contrib)[::-1][:top_n]
        top_terms = [feature_names[i] for i in top_indices if contrib[i] > 0]
        proba = proba_model.predict_proba(X[idx])[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        credibility_score = float(np.max(proba))
        alert = credibility_score < alert_threshold

        requests.append(
            UpdateOne(
                {"clean_text": clean_text},
                {"$set": {
                    "explanation_terms": top_terms,
                    "predicted_label": pred_label,
                    "credibility_score": credibility_score,
                    "alert": alert,
                    "explanation_text": (
                        f"This post is classified as '{pred_label}' with a credibility score "
                        f"of {credibility_score:.2f}. Dominant emotion: "
                        f"{post.get('dominant_emotion', 'unknown')}. "
                        f"Key terms: {', '.join(top_terms[:5]) if top_terms else 'none'}."
                    ),
                }},
                upsert=False,
            )
        )

    if requests:
        collection.bulk_write(requests, ordered=False)
        print("[Explainability] Per-post explanations stored in MongoDB.")


# ============================================================
# KMEANS + INTERPRETABILITY
# ============================================================

def train_kmeans(X, posts, vectorizer, n_clusters=8, cluster_by: str = "label_lang"):
    """
    Train KMeans and extract top TF-IDF terms per cluster.
    Save:
        - cluster assignments
        - cluster top terms
        - inertia metadata
    """
    os.makedirs("data/08_reporting", exist_ok=True)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)

    cluster_labels = model.labels_

    # Save global cluster assignments
    df_clusters = pd.DataFrame({
        "clean_text": [p["clean_text"] for p in posts],
        "cluster": cluster_labels
    })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    df_clusters.to_parquet(f"data/08_reporting/clusters_{timestamp}.parquet")

    # Extract top TF-IDF terms per global cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    cluster_keywords = {}

    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :15]]
        cluster_keywords[f"cluster_{i}"] = top_terms

    with open(f"data/08_reporting/cluster_keywords_{timestamp}.json", "w") as f:
        json.dump(cluster_keywords, f, indent=4)

    # Optional: group-specific clustering (per label and language)
    if cluster_by and cluster_by != "none":
        group_map = {}
        for idx, post in enumerate(posts):
            label = post.get("source_label", "unknown")
            lang = post.get("lang_detected", "unknown")
            if cluster_by == "label":
                key = label
            elif cluster_by == "lang":
                key = lang
            else:
                key = f"{label}__{lang}"
            group_map.setdefault(key, []).append(idx)

        group_keywords = {}
        group_assignments = []

        for key, indices in group_map.items():
            if len(indices) < 2:
                continue

            k = min(n_clusters, max(2, len(indices)))
            sub_X = X[indices]
            sub_model = KMeans(n_clusters=k, random_state=42, n_init=10)
            sub_model.fit(sub_X)

            order_centroids = sub_model.cluster_centers_.argsort()[:, ::-1]
            group_keywords[key] = {}
            for i in range(k):
                top_terms = [terms[ind] for ind in order_centroids[i, :15]]
                group_keywords[key][f"cluster_{i}"] = top_terms

            for local_idx, cluster_id in zip(indices, sub_model.labels_):
                group_assignments.append({
                    "clean_text": posts[local_idx]["clean_text"],
                    "group": key,
                    "cluster": int(cluster_id),
                })

        if group_assignments:
            df_group = pd.DataFrame(group_assignments)
            df_group.to_parquet(f"data/08_reporting/clusters_by_group_{timestamp}.parquet")

        with open(f"data/08_reporting/cluster_keywords_by_group_{timestamp}.json", "w") as f:
            json.dump(group_keywords, f, indent=4)

    print("[Clustering] Model trained and keywords extracted.")
    return model


# ============================================================
# CLASSIFICATION
# ============================================================

def train_classifier(X, labels):
    """
    Train Logistic Regression classifier.
    Save classification report and confusion matrix.
    """
    os.makedirs("data/08_reporting", exist_ok=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42  # now have a deterministic behavior
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with open(f"data/08_reporting/classification_report_{timestamp}.json", "w") as f:
        json.dump(report, f, indent=4)

    np.save(f"data/08_reporting/confusion_matrix_{timestamp}.npy", cm)

    print("[Classification] Report saved.")
    return clf, le


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_classifier(X, labels, classifier):
    """
    Calibrate probabilities with Platt scaling (sigmoid).
    """
    calibrated = CalibratedClassifierCV(classifier, method="sigmoid", cv=3)
    calibrated.fit(X, labels)
    print("[Calibration] Calibrated classifier trained.")
    return calibrated


# ============================================================
# PREDICTIONS OUTPUT
# ============================================================

def generate_predictions(posts: List[Dict], X, classifier, label_encoder, calibrated_classifier=None, alert_threshold: float = 0.7):
    """
    Save per-post predictions to data/07_model_output.
    """
    os.makedirs("data/07_model_output", exist_ok=True)

    model_for_proba = calibrated_classifier or classifier
    preds = classifier.predict(X)
    proba = model_for_proba.predict_proba(X)

    pred_labels = label_encoder.inverse_transform(preds)
    max_proba = proba.max(axis=1)

    rows = []
    for idx, post in enumerate(posts):
        post["predicted_label"] = pred_labels[idx]
        post["credibility_score"] = float(max_proba[idx])
        post["alert"] = bool(max_proba[idx] < alert_threshold)
        rows.append({
            "clean_text": post.get("clean_text"),
            "source_label": post.get("source_label"),
            "credibility_label": post.get("credibility_label", "unverified"),
            "predicted_label": pred_labels[idx],
            "credibility_score": float(max_proba[idx]),
            "alert": bool(max_proba[idx] < alert_threshold),
            "sentiment": post.get("sentiment", 0.0),
            "dominant_emotion": post.get("dominant_emotion", "unknown"),
            "emotion_scores": post.get("emotion_scores", {}),
        })

    df = pd.DataFrame(rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    df.to_parquet(f"data/07_model_output/predictions_{timestamp}.parquet")
    print("[Predictions] Saved model output.")


# ============================================================
# EXPLAINABILITY (CLASSIFIER FEATURES)
# ============================================================

def save_classifier_explanations(classifier, vectorizer, label_encoder, top_n: int = 15):
    """
    Save top TF-IDF terms per class based on Logistic Regression coefficients.
    """
    os.makedirs("data/08_reporting", exist_ok=True)

    if not hasattr(classifier, "coef_"):
        print("[Explainability] Classifier has no coef_. Skipping.")
        return

    feature_names = vectorizer.get_feature_names_out()
    explanations = {}

    coefs = classifier.coef_
    class_names = list(label_encoder.classes_)

    if coefs.shape[0] == 1 and len(class_names) == 2:
        coefs = np.vstack([-coefs[0], coefs[0]])

    for class_index, class_name in enumerate(class_names):
        class_coefs = coefs[class_index]
        top_indices = class_coefs.argsort()[::-1][:top_n]
        explanations[class_name] = [feature_names[i] for i in top_indices]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"data/08_reporting/classifier_top_terms_{timestamp}.json", "w") as f:
        json.dump(explanations, f, indent=4)

    print("[Explainability] Top terms saved.")


# ============================================================
# REPORTING FIGURES
# ============================================================

def generate_reporting_figures(_posts: Optional[List[Dict]] = None):
    """
    Create lightweight charts from the latest reporting files (if matplotlib is available).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Reporting] Matplotlib not available. Skipping figures.")
        return

    os.makedirs("data/08_reporting/figures", exist_ok=True)

    sentiment_files = sorted(glob.glob("data/08_reporting/sentiment_summary_*.json"))
    emotion_files = sorted(glob.glob("data/08_reporting/emotion_summary_*.json"))

    if sentiment_files:
        df_sent = pd.read_json(sentiment_files[-1])
        ax = df_sent["mean"].plot(kind="bar", title="Average sentiment per label")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("data/08_reporting/figures/sentiment_mean.png")
        plt.close(fig)

    if emotion_files:
        df_emotion = pd.read_json(emotion_files[-1])
        ax = df_emotion.plot(kind="bar", stacked=True, title="Emotion distribution per label")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("data/08_reporting/figures/emotion_distribution.png")
        plt.close(fig)

    print("[Reporting] Figures generated.")


# ============================================================
# TRANSFORMER MODEL (OPTIONAL)
# ============================================================

def train_transformer_model(
    posts: List[Dict],
    labels: List[str],
    enable: bool = False,
    model_name: Optional[str] = None,
    output_dir: str = "data/06_models/transformer",
    max_length: int = 128,
    num_train_epochs: int = 1,
    batch_size: int = 8,
):
    """
    Train a transformer classifier (optional). Requires transformers + datasets.
    """
    if not enable:
        print("[Transformer] Disabled by configuration.")
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset
    except Exception:
        print("[Transformer] transformers/datasets not installed. Skipping.")
        return None

    texts = [p.get("clean_text", "") for p in posts]
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    dataset = Dataset.from_dict({"text": texts, "label": numeric_labels})
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model_name = model_name or "cardiffnlp/twitter-xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)},
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    joblib.dump(label_encoder, os.path.join(output_dir, "transformer_label_encoder.joblib"))

    eval_metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=4)

    print(f"[Transformer] Model saved to {output_dir}")
    return output_dir


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(vectorizer, kmeans_model, classifier, label_encoder, calibrated_classifier=None):
    """
    Persist all trained models for reproducibility.
    """
    os.makedirs("data/06_models", exist_ok=True)

    joblib.dump(vectorizer, "data/06_models/vectorizer.joblib")
    joblib.dump(kmeans_model, "data/06_models/kmeans_model.joblib")
    joblib.dump(classifier, "data/06_models/classifier.joblib")
    joblib.dump(label_encoder, "data/06_models/label_encoder.joblib")
    if calibrated_classifier is not None:
        joblib.dump(calibrated_classifier, "data/06_models/calibrated_classifier.joblib")

    print("[Models] Saved successfully.")
