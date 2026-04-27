import sys, os, requests, time
from dotenv import load_dotenv
from app_config import load_app_config
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from getBlueskySearch import load_token, search_posts_backfill
from getBlueskyAuthorFeed import fetch_author_feed

# ---------- CONFIG ----------
load_dotenv()
app_config = load_app_config()

MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DATABASE_NAME")

SEARCH_QUERIES = {
    "science": "study",
    "ukraine": "conflict",
    "news": "breaking",
    "climate": "greenhouse effect"
}
"""
SEARCH_QUERIES_FR = {  # with some English because French returns very few results anyways...
    "science": "sciences OR recherche OR scientifique OR scientifiques OR étude OR études",
    "ukraine": "Ukraine OR ukrainien OR guerre OR conflit OR invasion OR war",
    "climate": "climat OR réchauffement climatique OR changement climatique or climate",
    "news": "news OR actus OR infos" 
}
"""
SEARCH_QUERIES_FR = {
    "science": "recherche",
    "ukraine": "guerre",
    "climate": "changement climatique",
    "news": "actualités" 
}

MAX_PAGES_PER_RUN = 5
LIMIT_PER_PAGE = 100
FR_LOW_ACTIVITY_THRESHOLD = 3  # number of consecutive low-FR runs before skipping
MAX_DOCS_PER_COLLECTION = int(app_config.get("max_docs_per_collection", 35000))
DATA_RETENTION_DAYS = int(app_config.get("data_retention_days", 0))
VERIFIED_HANDLES = app_config.get("verified_handles", [])
LANGS_FILTER = app_config.get("langs_filter", ["en", "fr"])
VERIFIED_AUTHOR_LIMIT_PER_PAGE = int(app_config.get("verified_author_limit_per_page", 50))
VERIFIED_AUTHOR_MAX_PAGES_PER_RUN = int(app_config.get("verified_author_max_pages_per_run", 1))

# ---------- DB ----------
def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def insert_posts(posts, collection_name):
    if not posts:
        print(f"[{collection_name}] No posts to insert.")
        return 0

    db = get_db()
    collection = db[collection_name]
    collection.create_index("uri", unique=True)
    collection.create_index("inserted_at")

    clean_posts = []
    skipped_null_uri = 0
    skipped_empty_text = 0

    for p in posts:
        if not p.get("uri"):
            skipped_null_uri += 1
            continue
        text = p.get("record", {}).get("text", "")
        if not str(text).strip():
            skipped_empty_text += 1
            continue
        p["inserted_at"] = datetime.now(timezone.utc)
        clean_posts.append(p)

    if not clean_posts:
        print(f"[{collection_name}] All posts skipped (missing URI).")
        return 0

    inserted_count = 0
    duplicates_count = 0

    try:
        result = collection.insert_many(clean_posts, ordered=False)
        inserted_count = len(result.inserted_ids)
        duplicates_count = len(clean_posts) - inserted_count
    except BulkWriteError as e:
        inserted_count = e.details.get("nInserted", 0)
        duplicates_count = len(clean_posts) - inserted_count
    except Exception as e:
        print(f"[{collection_name}] Unexpected insert error: {e}")
        return 0

    print(f"[{collection_name}] Inserted: {inserted_count}")
    print(f"[{collection_name}] Duplicates skipped: {duplicates_count}")
    print(f"[{collection_name}] Posts skipped (no URI): {skipped_null_uri}")
    print(f"[{collection_name}] Posts skipped (empty text): {skipped_empty_text}")

    cap = MAX_DOCS_PER_COLLECTION
    caps = app_config.get("collection_caps", {})
    if collection_name in caps:
        cap = int(caps[collection_name])

    prune_collection(collection, cap, DATA_RETENTION_DAYS)
    return inserted_count


def filter_author_posts(posts):
    allowed_langs = [l.strip() for l in LANGS_FILTER if isinstance(l, str) and l.strip()]
    filtered_posts = []

    for post in posts:
        if not post.get("uri"):
            continue

        langs = post.get("langs", [])
        if allowed_langs and langs and not any(lang in allowed_langs for lang in langs):
            continue

        text = post.get("record", {}).get("text", "")
        if not str(text).strip():
            continue

        filtered_posts.append(post)

    return filtered_posts


def prune_collection(collection, max_docs: int, retention_days: int):
    if retention_days and retention_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        deleted = collection.delete_many({"inserted_at": {"$lt": cutoff}})
        print(f"[{collection.name}] Retention cleanup deleted: {deleted.deleted_count}")

    if max_docs and max_docs > 0:
        total = collection.count_documents({})
        if total > max_docs:
            excess = total - max_docs
            old_ids = list(
                collection.find({}, {"_id": 1})
                .sort("inserted_at", 1)
                .limit(excess)
            )
            if old_ids:
                deleted = collection.delete_many({"_id": {"$in": [d["_id"] for d in old_ids]}})
                print(f"[{collection.name}] Max-docs cleanup deleted: {deleted.deleted_count}")

# ---------- MAIN ----------
def main():
    print("\n========== BLUESKY SEARCH COLLECTOR ==========")

    if not MONGO_URI or not DB_NAME:
        print("Mongo environment variables are not set. Exiting.")
        sys.exit(1)

    token = load_token()
    if not token:
        print("Authentication failed. Exiting.")
        sys.exit(1)

    db = get_db()
    state_coll = db["search_state"]
    total_inserted = 0

    for collection_name, query in SEARCH_QUERIES.items():
        print("\n--------------------------------------------")
        print(f"[COLLECTION] {collection_name}")
        print(f"[QUERY] {query}")

        state_doc = state_coll.find_one({"query_name": collection_name}) or {}
        total_posts = []

        pages_per_lang = max(1, MAX_PAGES_PER_RUN // 2)

        for lang in ["en", "fr"]:
            is_fr = lang == "fr"

            # Determine which query to use
            query_to_use = SEARCH_QUERIES_FR[collection_name] if is_fr else query

            last_cursor = state_doc.get(f"last_cursor_{lang}")
            exhausted = state_doc.get(f"exhausted_{lang}", False)
            fr_low_activity_counter = state_doc.get("fr_low_activity_counter", 0)

            # Skip French if low activity threshold reached
            if is_fr and fr_low_activity_counter >= FR_LOW_ACTIVITY_THRESHOLD:
                print(f"[INFO] {collection_name.upper()} - FR skipped due to low activity ({fr_low_activity_counter} consecutive low runs).")
                continue

            if last_cursor is None and not exhausted:
                print(f"[INFO] {collection_name.upper()} - {lang.upper()} search starting from scratch (no cursor).")

            max_pages = 1 if exhausted else pages_per_lang

            posts, new_cursor = search_posts_backfill(
                token,
                query=query_to_use,
                last_cursor=last_cursor,
                limit_per_page=LIMIT_PER_PAGE,
                max_pages=max_pages,
                lang=lang
            )

            total_posts.extend(posts)

            # Detect exhaustion
            if new_cursor is None and last_cursor is None and len(posts) <= 1:
                if is_fr:
                    fr_low_activity_counter += 1
                    print(f"[INFO] {collection_name.upper()} - FR low activity counter: {fr_low_activity_counter}")
                exhausted = True
            else:
                if is_fr:
                    fr_low_activity_counter = 0
                exhausted = False

            # Save cursor, exhausted, and FR counter
            update_doc = {
                f"last_cursor_{lang}": new_cursor,
                f"exhausted_{lang}": exhausted,
                "last_updated": datetime.now(timezone.utc)
            }
            if is_fr:
                update_doc["fr_low_activity_counter"] = fr_low_activity_counter

            state_coll.update_one(
                {"query_name": collection_name},
                {"$set": update_doc},
                upsert=True
            )

            print(f"[{collection_name}] Fetched {len(posts)} posts for {lang.upper()}")

        print(f"[{collection_name}] Total fetched this run: {len(total_posts)}")
        inserted = insert_posts(total_posts, collection_name)
        total_inserted += inserted
        print(f"[{collection_name}] Cursor(s) updated.\n")


    # ---------- VERIFIED NEWS FROM TRUSTED AUTHORS ----------
    feed_collection_name = "verified_news"
    verified_total_posts = []

    for handle in VERIFIED_HANDLES:
        state_key = f"{feed_collection_name}:{handle}"
        handle_state_doc = state_coll.find_one({"query_name": state_key}) or {}

        handle_posts, new_cursor = fetch_author_feed(
            token,
            handle,
            limit_per_page=VERIFIED_AUTHOR_LIMIT_PER_PAGE,
            max_pages=VERIFIED_AUTHOR_MAX_PAGES_PER_RUN,
            start_cursor=handle_state_doc.get("last_cursor")
        )

        handle_posts = filter_author_posts(handle_posts)
        verified_total_posts.extend(handle_posts)

        state_coll.update_one(
            {"query_name": state_key},
            {"$set": {"last_updated": datetime.now(timezone.utc), "last_cursor": new_cursor}},
            upsert=True
        )

        print(f"[{feed_collection_name}] {handle}: fetched {len(handle_posts)} posts")

    print(f"[{feed_collection_name}] Total fetched this run: {len(verified_total_posts)}")
    inserted = insert_posts(verified_total_posts, feed_collection_name)
    total_inserted += inserted
    print(f"[{feed_collection_name}] Trusted-author backfill complete.\n")

    print("============================================")
    print(f"RUN COMPLETE. Total new posts inserted: {total_inserted}")
    print("============================================\n")


if __name__ == "__main__":
    main()
