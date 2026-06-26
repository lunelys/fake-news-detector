import glob
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTING_DIR = PROJECT_ROOT / "bluesky-pipeline" / "data" / "08_reporting"

load_dotenv(PROJECT_ROOT / ".env")

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))
API_DEFAULT_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SOURCE_LABEL_OPTIONS = ["verified_news", "climate", "news", "ukraine", "science"]
LANGUAGE_OPTIONS = ["fr", "en"]
LATEST_POSTS_PREVIEW_LIMIT = 100
POST_FIELDS = {
    "clean_text": "Post text",
    "source_label": "Source",
    "credibility_label": "Reference label",
    "predicted_label": "Prediction",
    "credibility_score": "Confidence",
    "alert": "Alert",
    "sentiment": "Sentiment",
    "dominant_emotion": "Dominant emotion",
    "explanation_terms": "Explanation terms",
    "explanation_text": "Explanation",
}
DEFAULT_POSTS_PER_SOURCE = 2000
DEFAULT_CREDIBLE_TEXT = "EN DIRECT, canicule : après le pic de mercredi, Météo-France prévoit un jeudi « suffocant » et place 72 départements en vigilance rouge canicule"
DEFAULT_UNVERIFIED_TEXT = "Les canicules sont inventées par les médias pour contrôler la population et vendre des climatiseurs."
DEFAULT_BLUESKY_URL = "https://bsky.app/profile/polek.bsky.social/post/3mmpclzne4k2k"

ENERGY_UNITS = {
    "duration": "seconds",
    "emissions": "kg CO2eq",
    "emissions_rate": "kg CO2eq/s",
    "cpu_power": "watts",
    "gpu_power": "watts",
    "ram_power": "watts",
    "cpu_energy": "kWh",
    "gpu_energy": "kWh",
    "ram_energy": "kWh",
    "energy_consumed": "kWh",
    "ram_total_size": "GB",
    "pue": "ratio",
}

st.set_page_config(page_title="Thumalien Dashboard", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07040f;
            --panel: #110a1f;
            --panel-soft: #19102b;
            --border: #352052;
            --purple: #3F09C8;
            --purple-bright: #8b5cf6;
            --text: #f8f7ff;
            --muted: #c4b5fd;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #fb7185;
            --blue: #38bdf8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.22), transparent 28rem),
                linear-gradient(135deg, var(--bg) 0%, #0c0617 45%, #020103 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] {
            background: rgba(7, 4, 15, 0.86);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
            background: #05030a;
            border-right: 1px solid var(--border);
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        h1, h2, h3, h4, h5, h6,
        p, span, label, div, button {
            color: var(--text);
        }
        .stCaption, [data-testid="stCaptionContainer"], small {
            color: var(--muted) !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(17, 10, 31, 0.92);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
        }
        .status-card {
            background: rgba(17, 10, 31, 0.92);
            border: 1px solid var(--border);
            border-left: 5px solid var(--purple);
            border-radius: 8px;
            padding: 1rem;
            min-height: 112px;
        }
        .status-card strong {
            display: block;
            font-size: 0.86rem;
            color: var(--muted);
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .status-card span {
            display: block;
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--text);
        }
        .status-card small {
            display: block;
            margin-top: 0.35rem;
            color: var(--muted) !important;
        }
        .status-green {
            border-left-color: var(--green);
        }
        .status-amber {
            border-left-color: var(--amber);
        }
        .status-red {
            border-left-color: var(--red);
        }
        .status-blue {
            border-left-color: var(--blue);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text);
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--border);
        }
        button[data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px 8px 0 0;
            color: var(--muted);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(124, 58, 237, 0.24);
            color: var(--text);
            border-bottom: 2px solid var(--purple-bright);
        }
        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: var(--panel) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }
        .stMultiSelect [data-baseweb="tag"]:has(span[title="positive"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="trust"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="joy"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="anticip"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="anticipation"]) {
            background: rgba(52, 211, 153, 0.28) !important;
            color: #ecfdf5 !important;
            border: 1px solid rgba(52, 211, 153, 0.55) !important;
            border-radius: 999px !important;
        }
        .stMultiSelect [data-baseweb="tag"]:has(span[title="negative"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="fear"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="anger"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="sadness"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="disgust"]) {
            background: rgba(251, 113, 133, 0.28) !important;
            color: #fff1f2 !important;
            border: 1px solid rgba(251, 113, 133, 0.55) !important;
            border-radius: 999px !important;
        }
        .stMultiSelect [data-baseweb="tag"]:has(span[title="unknown"]),
        .stMultiSelect [data-baseweb="tag"]:has(span[title="surprise"]) {
            background: rgba(167, 139, 250, 0.24) !important;
            color: var(--text) !important;
            border: 1px solid rgba(167, 139, 250, 0.50) !important;
            border-radius: 999px !important;
        }
        .stMultiSelect [data-baseweb="tag"] span,
        .stMultiSelect [data-baseweb="tag"] svg {
            color: inherit !important;
        }
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: var(--purple-bright) !important;
            box-shadow: 0 0 0 1px var(--purple-bright) !important;
        }
        .stSlider [data-baseweb="slider"] > div {
            color: var(--purple-bright);
        }
        .stButton button {
            background: var(--purple);
            border: 1px solid var(--purple-bright);
            color: var(--text);
            border-radius: 8px;
            font-weight: 700;
        }
        .stButton button:hover {
            background: #3307a3;
            border-color: var(--text);
            color: var(--text);
        }
        div[data-testid="stAlert"] {
            background: rgba(25, 16, 43, 0.94);
            border: 1px solid var(--border);
            color: var(--text);
        }
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] div {
            color: var(--text);
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        pre, code {
            background: #05030a !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }
        details {
            background: rgba(17, 10, 31, 0.72);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
        }
        .stJson {
            background: var(--panel);
            border-radius: 8px;
        }
        .status-ok {
            color: var(--green);
            font-weight: 700;
        }
        .status-warn {
            color: var(--amber);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=10)
def call_predict(api_url: str, text: str, lang: str | None) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"text": text, "lang": lang or None},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60)
def fetch_bluesky_post_text(post_url: str) -> str:
    parsed = urlparse(post_url.strip())
    path_parts = [part for part in parsed.path.split("/") if part]

    if parsed.netloc not in {"bsky.app", "www.bsky.app"} or len(path_parts) < 4:
        raise ValueError("Use a Bluesky URL like https://bsky.app/profile/handle/post/post_id.")
    if path_parts[0] != "profile" or path_parts[2] != "post":
        raise ValueError("Use a Bluesky post URL from bsky.app/profile/.../post/...")

    handle = path_parts[1]
    post_id = path_parts[3]
    public_api = "https://public.api.bsky.app/xrpc"

    resolve_response = requests.get(
        f"{public_api}/com.atproto.identity.resolveHandle",
        params={"handle": handle},
        timeout=10,
    )
    resolve_response.raise_for_status()
    did = resolve_response.json()["did"]

    at_uri = f"at://{did}/app.bsky.feed.post/{post_id}"
    thread_response = requests.get(
        f"{public_api}/app.bsky.feed.getPostThread",
        params={"uri": at_uri},
        timeout=10,
    )
    thread_response.raise_for_status()
    thread = thread_response.json().get("thread", {})
    text = (
        thread.get("post", {})
        .get("record", {})
        .get("text", "")
        .strip()
    )

    if not text:
        raise ValueError("The post was found, but no text was available.")
    return text


@st.cache_data(ttl=10)
def call_health(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10)
def call_metrics(api_url: str) -> str:
    response = requests.get(f"{api_url.rstrip('/')}/metrics", timeout=5)
    response.raise_for_status()
    return response.text


@st.cache_data(ttl=30)
def load_clean_posts(source_labels: list[str], limit_per_source: int) -> tuple[pd.DataFrame, str | None]:
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DATABASE_NAME", "bluesky")
    collection_name = os.getenv("CLEAN_COLLECTION", "bluesky_posts_cleaned")

    if not mongo_url:
        return pd.DataFrame(), "MONGO_URL is not set."

    try:
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        collection = db[collection_name]
        frames = []
        for source_label in source_labels:
            query = {"source_label": source_label}
            projection = {"_id": 0}
            try:
                cursor = collection.find(query, projection).sort("inserted_at", -1).limit(limit_per_source)
                try:
                    cursor = cursor.allow_disk_use(True)
                except (AttributeError, TypeError):
                    pass
                source_frame = pd.DataFrame(list(cursor))
            except PyMongoError as exc:
                if getattr(exc, "code", None) != 292:
                    raise
                cursor = collection.find(query, projection).sort("_id", -1).limit(limit_per_source)
                source_frame = pd.DataFrame(list(cursor))
            if not source_frame.empty:
                frames.append(source_frame)

        if not frames:
            return pd.DataFrame(), None

        return pd.concat(frames, ignore_index=True), None
    except PyMongoError as exc:
        return pd.DataFrame(), f"MongoDB connection failed: {exc}"


def latest_file(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(REPORTING_DIR / pattern)))
    return Path(files[-1]) if files else None


def parse_prometheus_value(metrics_text: str, metric_name: str) -> float | None:
    total = 0.0
    found = False
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.startswith(metric_name):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
            found = True
        except (IndexError, ValueError):
            continue
    return total if found else None


def status_card(title: str, value: str, note: str, status: str) -> None:
    st.markdown(
        f"""
        <div class="status-card status-{status}">
            <strong>{title}</strong>
            <span>{value}</span>
            <small>{note}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_status(label: str, score: float, alert: bool) -> tuple[str, str, str]:
    if alert:
        return "Review", "amber", f"Confidence is below {ALERT_THRESHOLD:.0%}."
    if label == "credible":
        return "Pass", "green", "Predicted credible with enough confidence."
    return "Reject", "red", "Predicted unverified with enough confidence."


def sentiment_status(sentiment: float) -> tuple[str, str, str]:
    if sentiment <= -0.2:
        return "Negative", "red", "Below -0.20 sentiment."
    if sentiment >= 0.2:
        return "Positive", "green", "Above +0.20 sentiment."
    return "Neutral", "amber", "Between -0.20 and +0.20."


def gauge(
    title: str,
    value: float,
    minimum: float,
    maximum: float,
    steps: list[dict[str, Any]],
    suffix: str = "",
) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge",
            value=value,
            domain={"x": [0, 1], "y": [0, 0.88]},
            title={"text": title, "font": {"color": "#f8f7ff", "size": 18}},
            gauge={
                "axis": {"range": [minimum, maximum], "tickcolor": "#c4b5fd"},
                "bar": {"color": "#f8f7ff"},
                "bgcolor": "#110a1f",
                "bordercolor": "#352052",
                "steps": steps,
            },
        )
    )
    figure.add_annotation(
        x=0.5,
        y=0.36,
        text=f"{value:.1f}{suffix}",
        showarrow=False,
        font={"color": "#f8f7ff", "size": 34},
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="middle",
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8f7ff"},
        height=300,
        margin={"t": 62, "b": 28, "l": 22, "r": 22},
    )
    return figure


def confidence_gauge(score: float) -> go.Figure:
    return gauge(
        "Model confidence",
        score * 100,
        0,
        100,
        [
            {"range": [0, ALERT_THRESHOLD * 100], "color": "rgba(251, 191, 36, 0.48)"},
            {"range": [ALERT_THRESHOLD * 100, 100], "color": "rgba(52, 211, 153, 0.45)"},
        ],
        "%",
    )


def sentiment_gauge(sentiment: float) -> go.Figure:
    return gauge(
        "Sentiment",
        sentiment,
        -1,
        1,
        [
            {"range": [-1, -0.2], "color": "rgba(251, 113, 133, 0.50)"},
            {"range": [-0.2, 0.2], "color": "rgba(251, 191, 36, 0.42)"},
            {"range": [0.2, 1], "color": "rgba(52, 211, 153, 0.42)"},
        ],
    )


def probability_chart(probability_frame: pd.DataFrame, predicted_label: str) -> go.Figure:
    colors = [
        "#34d399" if label == "credible" else "#fb7185" if label == "unverified" else "#a78bfa"
        for label in probability_frame["label"]
    ]
    figure = go.Figure(
        go.Bar(
            x=probability_frame["label"],
            y=probability_frame["probability"],
            marker_color=colors,
            text=[f"{value:.1%}" for value in probability_frame["probability"]],
            textposition="auto",
        )
    )
    figure.update_layout(
        title=f"Class probabilities, predicted: {predicted_label}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,10,31,0.55)",
        font={"color": "#f8f7ff"},
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        height=300,
        margin={"t": 48, "b": 36, "l": 36, "r": 12},
    )
    return figure


def emotion_chart(emotion_scores: dict[str, float]) -> go.Figure:
    negative_emotions = {"fear", "anger", "negative", "sadness", "disgust"}
    positive_emotions = {"trust", "positive", "joy", "anticipation", "anticip"}
    emotion_df = pd.DataFrame(
        emotion_scores.items(),
        columns=["emotion", "score"],
    ).sort_values("score", ascending=False)
    colors = [
        "#fb7185" if emotion in negative_emotions else "#34d399" if emotion in positive_emotions else "#a78bfa"
        for emotion in emotion_df["emotion"]
    ]
    figure = go.Figure(
        go.Bar(
            x=emotion_df["score"],
            y=emotion_df["emotion"],
            orientation="h",
            marker_color=colors,
            text=[f"{score:.2f}" for score in emotion_df["score"]],
            textposition="auto",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,10,31,0.55)",
        font={"color": "#f8f7ff"},
        xaxis={"range": [0, max(0.01, emotion_df["score"].max() * 1.15)]},
        yaxis={"autorange": "reversed"},
        height=300,
        margin={"t": 20, "b": 32, "l": 82, "r": 12},
    )
    return figure


def sentiment_by_group_chart(frame: pd.DataFrame, group_column: str) -> go.Figure:
    sentiment = (
        frame.dropna(subset=[group_column, "sentiment"])
        .groupby(group_column)["sentiment"]
        .mean()
        .sort_values()
        .reset_index()
    )
    colors = [
        "#fb7185" if value <= -0.2 else "#34d399" if value >= 0.2 else "#fbbf24"
        for value in sentiment["sentiment"]
    ]
    figure = go.Figure(
        go.Bar(
            x=sentiment["sentiment"],
            y=sentiment[group_column],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f}" for value in sentiment["sentiment"]],
            textposition="auto",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,10,31,0.55)",
        font={"color": "#f8f7ff"},
        xaxis={"range": [-1, 1], "title": "Average sentiment (-1 to +1)"},
        height=320,
        margin={"t": 20, "b": 45, "l": 94, "r": 16},
    )
    return figure


def emotion_distribution_chart(frame: pd.DataFrame) -> go.Figure:
    negative_emotions = {"fear", "anger", "negative", "sadness", "disgust"}
    positive_emotions = {"trust", "positive", "joy", "anticipation", "anticip"}
    emotion_counts = frame["dominant_emotion"].fillna("unknown").value_counts().reset_index()
    emotion_counts.columns = ["emotion", "posts"]
    colors = [
        "#fb7185" if emotion in negative_emotions else "#34d399" if emotion in positive_emotions else "#a78bfa"
        for emotion in emotion_counts["emotion"]
    ]
    figure = go.Figure(
        go.Bar(
            x=emotion_counts["emotion"],
            y=emotion_counts["posts"],
            marker_color=colors,
            text=emotion_counts["posts"],
            textposition="auto",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,10,31,0.55)",
        font={"color": "#f8f7ff"},
        height=320,
        margin={"t": 18, "b": 40, "l": 36, "r": 12},
    )
    return figure


def emotion_category(emotion: Any) -> str:
    positive_emotions = {"trust", "positive", "joy", "anticipation", "anticip"}
    negative_emotions = {"fear", "anger", "negative", "sadness", "disgust"}
    value = str(emotion).lower()
    if value in positive_emotions:
        return "positive"
    if value in negative_emotions:
        return "negative"
    return "neutral"


def style_post_preview(frame: pd.DataFrame) -> Any:
    def style_emotion(value: Any) -> str:
        category = emotion_category(value)
        if category == "positive":
            return "background-color: rgba(52, 211, 153, 0.25); color: #ecfdf5;"
        if category == "negative":
            return "background-color: rgba(251, 113, 133, 0.25); color: #fff1f2;"
        return "background-color: rgba(167, 139, 250, 0.22); color: #f8f7ff;"

    def style_prediction(value: Any) -> str:
        if str(value).lower() == "credible":
            return "background-color: rgba(52, 211, 153, 0.22); color: #ecfdf5;"
        if str(value).lower() == "unverified":
            return "background-color: rgba(251, 113, 133, 0.22); color: #fff1f2;"
        return ""

    styler = frame.style
    if "Dominant emotion" in frame.columns:
        styler = styler.map(style_emotion, subset=["Dominant emotion"])
    if "Prediction" in frame.columns:
        styler = styler.map(style_prediction, subset=["Prediction"])
    if "Sentiment" in frame.columns:
        styler = styler.format({"Sentiment": "{:.3f}"})
    if "Confidence" in frame.columns:
        styler = styler.format({"Confidence": "{:.3f}"})
    return styler


def label_probabilities(probabilities: list[float], predicted_label: str) -> pd.DataFrame:
    if len(probabilities) == 2:
        labels = ["credible", "unverified"]
    else:
        labels = [f"class_{index}" for index in range(len(probabilities))]

    frame = pd.DataFrame({"label": labels, "probability": probabilities})
    if predicted_label in frame["label"].values:
        return frame

    best_index = int(frame["probability"].idxmax())
    frame.loc[best_index, "label"] = predicted_label
    return frame


def render_prediction() -> None:
    st.subheader("Analyze a Bluesky post")
    st.caption("Paste a post text, or fetch text from a public Bluesky post URL.")

    api_url = st.text_input(
        "FastAPI URL",
        API_DEFAULT_URL,
        help="Endpoint used by this tab to call the local FastAPI `/predict` service.",
    )
    post_url = st.text_input(
        "Bluesky post URL",
        DEFAULT_BLUESKY_URL,
        placeholder="https://bsky.app/profile/handle.bsky.social/post/...",
    )
    if "predict_text" not in st.session_state:
        st.session_state.predict_text = DEFAULT_CREDIBLE_TEXT

    if st.button("Fetch post text", use_container_width=True):
        if not post_url.strip():
            st.warning("Paste a Bluesky post URL first.")
        else:
            try:
                st.session_state.predict_text = fetch_bluesky_post_text(post_url)
                st.success("Post text fetched. You can now run the prediction.")
            except (requests.RequestException, KeyError, ValueError) as exc:
                st.error(f"Could not fetch Bluesky post: {exc}")

    example_left, example_right = st.columns(2)
    with example_left:
        if st.button("Use credible demo text", use_container_width=True):
            st.session_state.predict_text = DEFAULT_CREDIBLE_TEXT
    with example_right:
        if st.button("Use unverified demo text", use_container_width=True):
            st.session_state.predict_text = DEFAULT_UNVERIFIED_TEXT

    text = st.text_area(
        "Post text",
        key="predict_text",
        height=150,
        help="Paste text manually, use a demo button, or fetch public text from the Bluesky URL field.",
    )
    lang = st.selectbox("Language hint", ["fr", "en", "auto"], index=0)

    if st.button("Run prediction", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Add some text before running the model.")
            return

        try:
            result = call_predict(api_url, text.strip(), None if lang == "auto" else lang)
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            return

        label = result["predicted_label"]
        score = result["credibility_score"]
        alert = result["alert"]
        sentiment = result["sentiment"]
        verdict, verdict_status, verdict_note = prediction_status(label, score, alert)
        sentiment_label, sentiment_color, sentiment_note = sentiment_status(sentiment)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_card("Decision", verdict, verdict_note, verdict_status)
        with col2:
            status_card("Prediction", label, "Classifier output.", verdict_status)
        with col3:
            status_card("Sentiment", sentiment_label, sentiment_note, sentiment_color)
        with col4:
            status_card("Alert", "Yes" if alert else "No", f"Review threshold: {ALERT_THRESHOLD:.0%}.", "amber" if alert else "green")

        if alert:
            st.warning("This is below the review threshold, so treat it as unsure and review it manually.")
        else:
            st.success("The model is confident enough to avoid an uncertainty alert.")

        st.markdown("<div style='height: 0.85rem'></div>", unsafe_allow_html=True)
        gauge_left, gauge_right = st.columns(2)
        with gauge_left:
            st.plotly_chart(confidence_gauge(score), use_container_width=True)
        with gauge_right:
            st.plotly_chart(sentiment_gauge(sentiment), use_container_width=True)

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("**Class probabilities**")
            prob_df = label_probabilities(result["probabilities"], label)
            st.plotly_chart(probability_chart(prob_df, label), use_container_width=True)

            st.markdown("**Explanation**")
            st.write(result["explanation_text"])
            st.write("Key terms:", ", ".join(result["top_terms"]) if result["top_terms"] else "none")

        with right:
            st.markdown("**Emotion signals**")
            emotion_scores = {
                key: value
                for key, value in result["emotion_scores"].items()
                if isinstance(value, (int, float)) and value > 0
            }
            if emotion_scores:
                st.plotly_chart(emotion_chart(emotion_scores), use_container_width=True)
                st.caption(f"Dominant emotion: {result['dominant_emotion']}")
            else:
                st.info("No emotion lexicon hit was found, so dominant emotion is unknown.")

        with st.expander("Raw API response"):
            st.json(result)


def render_posts() -> None:
    st.subheader("Stored scored posts")
    st.caption(
        "This tab reads the latest cleaned/scored documents from MongoDB. "
    )

    selected_sources = st.multiselect(
        "Source labels",
        SOURCE_LABEL_OPTIONS,
        default=SOURCE_LABEL_OPTIONS,
        help="Project source groups stored as `source_label` in the cleaned MongoDB collection.",
    )
    limit_per_source = st.slider(
        "Analysis sample per source",
        100,
        10000,
        DEFAULT_POSTS_PER_SOURCE,
        step=100,
        help=(
            "Loads this many recent cleaned/scored posts per selected source. "
            "This avoids MongoDB sort memory errors on large collections."
        ),
    )

    with st.expander("How to read this tab"):
        st.write(
            "The stored posts view is historical data from the Kedro pipeline, not a live API prediction. "
            "Use it to inspect what has already been collected, cleaned, enriched, and scored."
        )
        st.write(
            "Charts use the loaded recent sample for each selected source. "
            "This keeps MongoDB queries responsive and avoids sort memory limits on large collections."
        )
        st.write(
            "The source selector queries the cleaned collection for each selected project source. "
            "If a source returns no rows, that source probably has not been scored into `bluesky_posts_cleaned` yet."
        )
        st.write(
            "Only French and English are kept here because the project scope is EN/FR. "
            "Other language codes can exist in raw social data, but they are hidden from this dashboard view."
        )

    if not selected_sources:
        st.warning("Select at least one source label.")
        return

    df, error = load_clean_posts(selected_sources, limit_per_source)

    if error:
        st.warning(error)
        st.caption("Prediction still works without MongoDB as long as the FastAPI service is running.")
        return

    if df.empty:
        st.info("No cleaned posts found yet. Run the Kedro scoring pipeline when you want historical dashboard data.")
        return

    if "lang_detected" in df.columns:
        df = df[df["lang_detected"].isin(LANGUAGE_OPTIONS)]

    filtered = df.copy()
    emotions = sorted(filtered.get("dominant_emotion", pd.Series(dtype=str)).dropna().unique().tolist())
    selected_emotions = st.multiselect(
        "Dominant emotion",
        emotions,
        default=emotions,
        help="Filters posts by strongest detected emotion. `unknown` means no emotion signal was found.",
    )
    if selected_emotions and "dominant_emotion" in filtered.columns:
        filtered = filtered[filtered["dominant_emotion"].isin(selected_emotions)]

    if filtered.empty:
        st.info("No EN/FR scored posts match the selected sources and emotion filters.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Posts",
        len(filtered),
        help="Number of loaded cleaned/scored EN/FR posts matching the current filters.",
    )
    col2.metric(
        "Avg sentiment",
        round(filtered.get("sentiment", pd.Series([0])).mean(), 3),
        help="Mean sentiment polarity from -1 negative to +1 positive.",
    )
    col3.metric(
        "Labels",
        filtered.get("source_label", pd.Series(dtype=str)).nunique(),
        help="Number of selected source groups that currently have matching scored posts.",
    )

    chart_left, chart_right = st.columns(2)
    if "source_label" in filtered.columns and "sentiment" in filtered.columns:
        with chart_left:
            st.subheader("Sentiment by source")
            st.caption("Average sentiment per source label. Red is negative, orange is neutral, green is positive.")
            st.plotly_chart(sentiment_by_group_chart(filtered, "source_label"), use_container_width=True)
    if "lang_detected" in filtered.columns and "sentiment" in filtered.columns:
        with chart_right:
            st.subheader("Sentiment by language")
            st.caption("Average sentiment for French and English posts only.")
            st.plotly_chart(sentiment_by_group_chart(filtered, "lang_detected"), use_container_width=True)

    if "dominant_emotion" in filtered.columns:
        st.subheader("Emotion distribution")
        st.caption(
            "Counts posts by their strongest detected emotion after filtering. "
            "`unknown` means no emotion signal was found or emotion enrichment failed."
        )
        st.plotly_chart(emotion_distribution_chart(filtered), use_container_width=True)

    columns = [
        column
        for column in [
            "clean_text",
            "source_label",
            "credibility_label",
            "predicted_label",
            "credibility_score",
            "alert",
            "sentiment",
            "dominant_emotion",
            "explanation_terms",
            "explanation_text",
        ]
        if column in filtered.columns
    ]

    st.subheader("Latest scored posts")
    st.caption(f"Preview capped at the {LATEST_POSTS_PREVIEW_LIMIT} most recent rows after filters.")
    if "inserted_at" in filtered.columns:
        filtered = filtered.sort_values("inserted_at", ascending=False)
    preview = filtered[columns].head(LATEST_POSTS_PREVIEW_LIMIT).rename(columns=POST_FIELDS)
    st.dataframe(style_post_preview(preview), use_container_width=True)


def render_monitoring() -> None:
    st.subheader("API monitoring")
    st.caption("This tab checks whether the FastAPI service is reachable and summarizes Prometheus runtime metrics.")
    api_url = st.text_input(
        "FastAPI URL",
        API_DEFAULT_URL,
        key="monitoring_api_url",
        help="Endpoint used by this tab to call `/health` and `/metrics`.",
    )

    with st.expander("How to read monitoring"):
        st.write("`GET /health` confirms the API process is alive and can respond.")
        st.write("`GET /metrics` exposes Prometheus counters and histograms. Request count is cumulative since the API started.")
        st.write("Average latency is estimated as latency sum divided by latency count, in seconds.")

    try:
        health = call_health(api_url)
        st.markdown(f"<span class='status-ok'>API healthy</span> `{health}`", unsafe_allow_html=True)
    except requests.RequestException as exc:
        st.markdown("<span class='status-warn'>API unavailable</span>", unsafe_allow_html=True)
        st.caption(str(exc))

    try:
        metrics = call_metrics(api_url)
        requests_total = parse_prometheus_value(metrics, "api_requests_total")
        latency_count = parse_prometheus_value(metrics, "api_request_latency_seconds_count")
        latency_sum = parse_prometheus_value(metrics, "api_request_latency_seconds_sum")
        avg_latency = latency_sum / latency_count if latency_count and latency_sum is not None else None

        col1, col2, col3 = st.columns(3)
        with col1:
            status_card("API requests", str(int(requests_total or 0)), "Cumulative calls since API start.", "blue")
        with col2:
            status_card(
                "Avg latency",
                f"{avg_latency:.3f}s" if avg_latency is not None else "n/a",
                "Lower is better; unit is seconds.",
                "green" if avg_latency is not None and avg_latency < 1 else "amber",
            )
        with col3:
            status_card("Metrics source", "/metrics", "Prometheus text exposition.", "blue")
        
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        
        with st.expander("Raw Prometheus metrics"):
            st.code(metrics, language="text")
    except requests.RequestException as exc:
        st.info(f"Metrics unavailable: {exc}")


def render_green_it() -> None:
    st.subheader("Green IT")
    st.caption("This tab summarizes the latest CodeCarbon energy report generated by the Kedro energy runner.")

    with st.expander("How to read Green IT metrics"):
        st.write("Power columns are measured in watts. Energy columns are measured in kilowatt-hours.")
        st.write("`emissions` is the estimated total carbon footprint in kg CO2 equivalent.")
        st.write("`emissions_rate` is kg CO2 equivalent per second, so very small local runs often display as `0.0000` when rounded.")
        st.write("These values are estimates for comparing runs and documenting Green IT effort; they are not billing-grade measurements.")

    energy_path = latest_file("energy_report_*.csv")
    if not energy_path:
        st.info("No CodeCarbon energy report found yet.")
        return

    energy = pd.read_csv(energy_path)
    st.caption(f"Latest report: {energy_path.relative_to(PROJECT_ROOT)}")
    latest = energy.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_card("Emissions", f"{latest.get('emissions', 0):.6f}", "kg CO2eq total.", "green")
    with col2:
        status_card("Energy", f"{latest.get('energy_consumed', 0):.4f}", "kWh consumed.", "green")
    with col3:
        status_card("Duration", f"{latest.get('duration', 0):.0f}", "seconds tracked.", "blue")
    with col4:
        status_card("Emissions rate", f"{latest.get('emissions_rate', 0):.8f}", "kg CO2eq/s.", "green")

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    display_columns = [column for column in ENERGY_UNITS if column in energy.columns]
    display = energy[display_columns].tail(10).copy()
    display.columns = [
        f"{column} ({ENERGY_UNITS[column]})"
        for column in display.columns
    ]
    st.dataframe(display, use_container_width=True, height=120)


inject_styles()

st.title("Credibility Analyzer for Bluesky Posts")
st.markdown(
    "By Malo & Lunelys for the Mastère 1 Big Data & Ia at Sup de Vinci Rennes. " \
    "NLP-Powered Credibility Scoring Dashboard with live credibility analysis, "
    "model explanations, operational monitoring, and Green IT evidence. "
    "Code available at [lunelys/fake-news-detector](https://github.com/lunelys/fake-news-detector)"
)

predict_tab, posts_tab, monitoring_tab, green_it_tab = st.tabs(
    ["Predict", "Stored posts", "Monitoring", "Green IT"]
)

with predict_tab:
    render_prediction()

with posts_tab:
    render_posts()

with monitoring_tab:
    render_monitoring()

with green_it_tab:
    render_green_it()
