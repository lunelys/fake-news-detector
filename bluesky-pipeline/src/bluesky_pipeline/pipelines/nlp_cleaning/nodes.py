from pymongo import MongoClient
from typing import List, Dict
from unidecode import unidecode
import re
import spacy
import numpy as np
from langdetect import detect
from sqlalchemy import create_engine, text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

import joblib
import os
import json
import pandas as pd
from datetime import datetime, timezone


# ============================================================
# LOAD SPACY MODELS
# ============================================================

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


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

        langs = post.get("langs", [])
        lang = langs[0] if langs else None

        if not lang:
            try:
                lang = detect(text)
            except:
                lang = "en"

        doc = nlp_en(text) if lang.startswith("en") else nlp_fr(text)

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
# TF-IDF VECTORIZATION
# ============================================================

def vectorize_posts(posts: List[Dict], max_features: int = 5000):
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

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
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
        blob = TextBlob(post["clean_text"])
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
# STORE TO POSTGRES
# ============================================================

def store_vectors_to_postgres(posts, X, postgres_uri, batch_size=1000):
    """
    Store TF-IDF embeddings in PostgreSQL using pgvector.
    Idempotent via ON CONFLICT on clean_text.
    """
    engine = create_engine(postgres_uri)

    stmt = text("""
        INSERT INTO posts_vectors (clean_text, embedding, label, sentiment)
        VALUES (:text, :vector, :label, :sentiment)
        ON CONFLICT (clean_text) DO NOTHING
    """)

    with engine.begin() as conn:
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            batch_posts = posts[start:end]
            batch_vectors = X[start:end]

            data = []
            for i, post in enumerate(batch_posts):
                vec_list = batch_vectors[i].toarray()[0].tolist()
                data.append({
                    "text": post["clean_text"],
                    "vector": vec_list,
                    "label": post["source_label"],
                    "sentiment": post.get("sentiment", 0.0)
                })

            conn.execute(stmt, data)

    print("[Postgres] Storage complete.")


# ============================================================
# KMEANS + INTERPRETABILITY
# ============================================================

def train_kmeans(X, posts, vectorizer, n_clusters=5):
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

    # Save cluster assignments
    df_clusters = pd.DataFrame({
        "clean_text": [p["clean_text"] for p in posts],
        "cluster": cluster_labels
    })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    df_clusters.to_parquet(f"data/08_reporting/clusters_{timestamp}.parquet")

    # Extract top TF-IDF terms per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    cluster_keywords = {}

    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :15]]
        cluster_keywords[f"cluster_{i}"] = top_terms

    with open(f"data/08_reporting/cluster_keywords_{timestamp}.json", "w") as f:
        json.dump(cluster_keywords, f, indent=4)

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
# SAVE MODELS
# ============================================================

def save_models(vectorizer, kmeans_model, classifier, label_encoder):
    """
    Persist all trained models for reproducibility.
    """
    os.makedirs("data/06_models", exist_ok=True)

    joblib.dump(vectorizer, "data/06_models/vectorizer.joblib")
    joblib.dump(kmeans_model, "data/06_models/kmeans_model.joblib")
    joblib.dump(classifier, "data/06_models/classifier.joblib")
    joblib.dump(label_encoder, "data/06_models/label_encoder.joblib")

    print("[Models] Saved successfully.")