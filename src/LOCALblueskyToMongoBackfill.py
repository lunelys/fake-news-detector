import sys, os, requests, time
from dotenv import load_dotenv
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from getBlueskySearch import load_token, search_posts_backfill, get_verified_news_feed

sys.stdout = open(r"C:\bluesky_log.txt", "a")
sys.stderr = sys.stdout
print("---- Script started ----")


# ---------- CONFIG ----------
load_dotenv("C:\\Users\\lunel\\OneDrive - SUP DE VINCI\\Documents\\Mastère1\\Projet_d_etude_G7\\.env")


MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DATABASE_NAME")

SEARCH_QUERIES = {
    "science": "science OR research",
    "ukraine": "Ukraine OR Ukrainian OR war",
    "news": "news",
    "climate": "climate OR global warming"
}

SEARCH_QUERIES_FR = {  # with some English because French returns very few results anyways...
    "science": "sciences OR recherche OR scientifique OR scientifiques OR étude OR études",
    "ukraine": "Ukraine OR ukrainien OR guerre OR conflit OR invasion OR war",
    "climate": "climat OR réchauffement climatique OR changement climatique or climate",
    "news": "news" 
}

MAX_PAGES_PER_RUN = 5
LIMIT_PER_PAGE = 100
FR_LOW_ACTIVITY_THRESHOLD = 3  # number of consecutive low-FR runs before skipping

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

    clean_posts = []
    skipped_null_uri = 0

    for p in posts:
        if not p.get("uri"):
            skipped_null_uri += 1
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
    return inserted_count

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

    # ---------- VERIFIED NEWS ----------
    feed_collection_name = "verified_news"
    feed_state_doc = state_coll.find_one({"query_name": feed_collection_name}) or {}

    verified_posts, _ = get_verified_news_feed(
        token,
        limit_per_page=100,
        max_pages=1,
        start_cursor=None
    )

    print(f"[{feed_collection_name}] Total fetched this run: {len(verified_posts)}")
    inserted = insert_posts(verified_posts, feed_collection_name)
    total_inserted += inserted

    # Always fetch latest; no cursor needed
    state_coll.update_one(
        {"query_name": feed_collection_name},
        {"$set": {"last_updated": datetime.now(timezone.utc)}},
        upsert=True
    )
    print(f"[{feed_collection_name}] Latest posts fetched.\n")

    print("============================================")
    print(f"RUN COMPLETE. Total new posts inserted: {total_inserted}")
    print("============================================\n")


if __name__ == "__main__":
    main()