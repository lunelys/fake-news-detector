import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

BLUESKY_USERNAME = os.getenv("BSKY_IDENTIFIER")
BLUESKY_PASSWORD = os.getenv("BSKY_PASSWORD")
API_URL = "https://bsky.social/xrpc"

def load_token():
    print("[AUTH] Requesting Bluesky session token...")
    payload = {
        "identifier": BLUESKY_USERNAME,
        "password": BLUESKY_PASSWORD
    }

    r = requests.post(f"{API_URL}/com.atproto.server.createSession", json=payload)
    if r.status_code != 200:
        print("[AUTH] Authentication failed:", r.text)
        return None

    print("[AUTH] Token successfully retrieved.")
    return r.json().get("accessJwt")

def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}

def search_posts_backfill(
    token,
    query,
    last_cursor=None,
    limit_per_page=100,
    max_pages=20,
    sleep_sec=0.5,
    lang=None,
):
    """
    Historical search-based backfill using app.bsky.feed.searchPosts.
    Handles pagination until max_pages or end of feed.
    """
    print(f"\n[SEARCH] Starting query: '{query}'")
    print(f"[SEARCH] Starting from cursor: {last_cursor}")

    all_posts = []
    cursor = last_cursor

    for page in range(1, max_pages + 1):
        params = {"q": query, "limit": limit_per_page, "sort": "latest"}
        if cursor:
            params["cursor"] = cursor
        if lang:
            params["lang"] = lang 

        r = requests.get(
            f"{API_URL}/app.bsky.feed.searchPosts",
            headers=_auth_headers(token),
            params=params
        )

        if r.status_code == 429:
            print("[RATE LIMIT] Hit 429. Sleeping 60 seconds...")
            time.sleep(60)
            continue

        if r.status_code != 200:
            print(f"[ERROR] Page {page} failed:", r.text)
            break

        data = r.json()
        posts = data.get("posts", [])
        print(f"[SEARCH] Page {page}: {len(posts)} posts returned.")

        if not posts:
            print("[SEARCH] No more posts available. Stopping.")
            break

        all_posts.extend(posts)
        cursor = data.get("cursor")

        if not cursor:
            print("[SEARCH] No cursor returned. Reached end.")
            break

        time.sleep(sleep_sec)

    print(f"[SEARCH] Finished query '{query}'. Total fetched: {len(all_posts)}")
    return all_posts, cursor


def get_verified_news_feed(token, limit_per_page=100, max_pages=1, start_cursor=None, sleep_sec=0.5, use_cursor: bool = False):
    """
    Fetch latest posts from 'Verified News'.
    Only 1 page per run; cursor ignored to always get most recent posts.
    """
    feed_uri = "at://did:plc:kkf4naxqmweop7dv4l2iqqf5/app.bsky.feed.generator/verified-news"

    print(f"\n[FEED] Starting 'Verified News' feed: {feed_uri}")

    all_posts = []

    headers = _auth_headers(token)
    headers["Accept-Language"] = "en,fr"

    params = {"feed": feed_uri, "limit": limit_per_page}
    if use_cursor and start_cursor:
        params["cursor"] = start_cursor

    r = requests.get(f"{API_URL}/app.bsky.feed.getFeed", headers=headers, params=params)

    if r.status_code == 429:
        print("[RATE LIMIT] Hit 429. Sleeping 60 seconds...")
        time.sleep(60)
        r = requests.get(f"{API_URL}/app.bsky.feed.getFeed", headers=headers, params=params)

    if r.status_code != 200:
        print(f"[ERROR] Feed request failed:", r.text)
        return all_posts, None

    data = r.json()
    posts = data.get("feed", [])

    for p in posts:
        post_obj = p.get("post", p)
        langs = post_obj.get("langs", [])
        if langs and not any(lang in ["en", "fr"] for lang in langs):
            continue
        all_posts.append(post_obj)

    print(f"[FEED] Finished 'Verified News'. Total fetched: {len(all_posts)}")
    return all_posts, data.get("cursor") if use_cursor else None
