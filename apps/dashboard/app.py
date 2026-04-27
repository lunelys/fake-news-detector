import os
import glob
import pandas as pd
import streamlit as st
from pymongo import MongoClient


st.set_page_config(page_title="Thumalien Dashboard", layout="wide")


def get_db():
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DATABASE_NAME", "bluesky")
    if not mongo_url:
        st.error("MONGO_URL is not set.")
        st.stop()
    client = MongoClient(mongo_url)
    return client[db_name]


def load_clean_posts(limit=2000):
    db = get_db()
    collection = db[os.getenv("CLEAN_COLLECTION", "bluesky_posts_cleaned")]
    cursor = collection.find({}, {"_id": 0}).sort("inserted_at", -1).limit(limit)
    return pd.DataFrame(list(cursor))


def latest_report(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


st.title("Thumalien - Fake News Detection Dashboard")

df = load_clean_posts()
if df.empty:
    st.warning("No cleaned posts found. Run the Kedro pipeline first.")
    st.stop()

st.sidebar.header("Filters")
labels = sorted(df["source_label"].dropna().unique().tolist())
emotions = sorted(df["dominant_emotion"].dropna().unique().tolist())
langs = sorted(df.get("lang_detected", pd.Series(dtype=str)).dropna().unique().tolist())

selected_labels = st.sidebar.multiselect("Source label", labels, default=labels)
selected_emotions = st.sidebar.multiselect("Dominant emotion", emotions, default=emotions)
selected_langs = st.sidebar.multiselect("Language", langs, default=langs if langs else [])

filtered = df[
    df["source_label"].isin(selected_labels)
    & df["dominant_emotion"].isin(selected_emotions)
]

if selected_langs and "lang_detected" in filtered.columns:
    filtered = filtered[filtered["lang_detected"].isin(selected_langs)]

col1, col2, col3 = st.columns(3)
col1.metric("Posts (filtered)", len(filtered))
col2.metric("Avg sentiment", round(filtered["sentiment"].mean(), 3))
col3.metric("Unique labels", filtered["source_label"].nunique())

st.subheader("Emotion distribution")
emotion_counts = filtered["dominant_emotion"].value_counts()
st.bar_chart(emotion_counts)

st.subheader("Latest cleaned posts")
columns = ["clean_text", "source_label", "sentiment", "dominant_emotion"]
if "credibility_label" in filtered.columns:
    columns.append("credibility_label")
if "credibility_score" in filtered.columns:
    columns.append("credibility_score")
if "alert" in filtered.columns:
    columns.append("alert")
if "emotion_scores" in filtered.columns:
    columns.append("emotion_scores")
if "explanation_terms" in filtered.columns:
    columns.append("explanation_terms")
if "explanation_text" in filtered.columns:
    columns.append("explanation_text")

st.dataframe(
    filtered[columns].head(200),
    use_container_width=True,
)

st.subheader("Latest reporting outputs")
sent_path = latest_report("bluesky-pipeline/data/08_reporting/sentiment_summary_*.json")
emo_path = latest_report("bluesky-pipeline/data/08_reporting/emotion_summary_*.json")

if sent_path:
    st.caption(f"Sentiment summary: {sent_path}")
    st.json(pd.read_json(sent_path).to_dict())

if emo_path:
    st.caption(f"Emotion summary: {emo_path}")
    st.json(pd.read_json(emo_path).to_dict())
