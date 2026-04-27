from getBlueskyFeed import (
    load_token,
    get_ukrainian_feed,
    get_science_feed,
    get_hot_feed,
    get_verified_news_feed,
)
from mongoConnect import get_db
from datetime import datetime, timezone, timedelta
from pymongo.errors import BulkWriteError
from app_config import load_app_config


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


def insert_feed_data(feed_data, collection_name):
    if not feed_data:
        print(f"No data to insert into '{collection_name}'.")
        return

    db = get_db()
    collection = db[collection_name]

    # Ensure unique index on post.uri (avoid duplicates)
    collection.create_index([("post.uri", 1)], unique=True)
    collection.create_index("inserted_at")

    filtered = []
    for item in feed_data:
        text = item.get("post", {}).get("record", {}).get("text", "")
        if not str(text).strip():
            continue
        item["inserted_at"] = datetime.now(timezone.utc)
        filtered.append(item)

    try:
        result = collection.insert_many(filtered)
        print(f"{len(result.inserted_ids)} documents inserted into '{collection_name}'.")
    except BulkWriteError as e:
        inserted = e.details.get("nInserted", 0)
        print(f"{inserted} new documents inserted into '{collection_name}' (duplicates ignored).")
    except Exception as e:
        print(f"Unexpected error in '{collection_name}': {e}")
    finally:
        config = load_app_config()
        caps = config.get("collection_caps", {})
        max_docs = int(caps.get(collection_name, config.get("max_docs_per_collection", 35000)))
        retention_days = int(config.get("data_retention_days", 0))
        prune_collection(collection, max_docs, retention_days)


def main():
    token = load_token()
    if not token:
        print("Token not loaded. Check credentials or env variables.")
        return

    feeds = [
        ("hot", get_hot_feed),
        ("ukrainian", get_ukrainian_feed),
        ("science", get_science_feed),
        ("verified_news", get_verified_news_feed),
    ]

    for name, feed_func in feeds:
        print(f"Fetching {name}...")
        posts = feed_func(token, limit=50)  # 50 posts per page
        print(f"{len(posts)} posts fetched from {name} feed.")
        insert_feed_data(posts, name)


if __name__ == "__main__":
    main()
