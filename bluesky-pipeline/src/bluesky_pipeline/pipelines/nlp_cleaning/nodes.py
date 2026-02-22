from pymongo import MongoClient
from typing import List, Dict
from unidecode import unidecode
import re
import spacy

# Load spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


# --------------------------
# MongoDB loader (merging multiple collections)
# --------------------------
def load_raw_posts(mongo_uri: str, db_name: str, raw_collections: List[str]) -> List[Dict]:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    all_posts = []
    for col_name in raw_collections:
        all_posts.extend(list(db[col_name].find({})))
    return all_posts


def extract_text(post: dict) -> str:
    """Extract main text from a Bluesky post."""
    # Try main record text
    text = post.get("record", {}).get("text")
    if text:
        return text
    # Fallback: check for nested embed record text
    embed_record = post.get("record", {}).get("embed", {}).get("record", {}).get("record", {})
    return embed_record.get("text", "") or ""

# --------------------------
# Text cleaning
# --------------------------
def clean_text_node(posts: list[dict], text_field: str = None) -> list[dict]:
    cleaned = []
    for post in posts:
        # Use our helper instead of post.get("text")
        text = extract_text(post)
        if not text:
            continue
        text = text.lower()                        # lowercase
        text = re.sub(r"http\S+", "", text)        # remove URLs
        text = re.sub(r"@\w+", "", text)           # remove mentions
        text = re.sub(r"#\w+", "", text)           # remove hashtags
        text = re.sub(r"[^\w\s]", "", text)        # remove punctuation
        text = unidecode(text)                     # remove accents/emojis
        post["clean_text"] = text.strip()
        cleaned.append(post)
    return cleaned


# --------------------------
# Tokenization + Lemmatization
# --------------------------
def tokenize_and_lemmatize(posts: List[Dict], lang_field: str = "lang", text_field: str = "clean_text") -> List[Dict]:
    processed = []
    for post in posts:
        text = post.get(text_field, "")
        if not text:
            continue
        lang = post.get(lang_field, "en")
        doc = nlp_en(text) if lang == "en" else nlp_fr(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        post["tokens"] = tokens
        processed.append(post)
    return processed


def debug_clean_posts(posts: list):
    print(f"[DEBUG] Number of cleaned posts: {len(posts)}")
    if posts:
        print("Sample post:", posts[0])
    return posts


# --------------------------
# Save cleaned data back to MongoDB
# --------------------------
def store_clean_posts(posts: List[Dict], mongo_uri: str, db_name: str, clean_collection: str) -> int:
    if not posts:
        print("No posts to insert!")
        return 0

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[clean_collection]
    collection.create_index("uri", unique=True)  # avoid duplicates

    inserted_count = 0
    for post in posts:
        try:
            collection.insert_one(post)
            inserted_count += 1
        except Exception as e:
            print(f"Skipping post due to error: {e}")
            continue

    print(f"Inserted {inserted_count} posts into {clean_collection}")
    return inserted_count