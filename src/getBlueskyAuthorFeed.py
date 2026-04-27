import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BLUESKY_USERNAME = os.getenv("BSKY_IDENTIFIER")
BLUESKY_PASSWORD = os.getenv("BSKY_PASSWORD")

API_URL = "https://bsky.social/xrpc"


def load_token():
    payload = {
        "identifier": BLUESKY_USERNAME,
        "password": BLUESKY_PASSWORD,
    }
    r = requests.post(f"{API_URL}/com.atproto.server.createSession", json=payload)
    if r.status_code != 200:
        print("Authentication error:", r.text)
        return None
    return r.json().get("accessJwt")


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def fetch_author_feed(token, handle, limit_per_page=100, max_pages=5, sleep_sec=0.4, start_cursor=None):
    """
    Fetch posts for a specific actor handle.
    """
    all_posts = []
    cursor = start_cursor

    for page in range(max_pages):
        params = {"actor": handle, "limit": limit_per_page}
        if cursor:
            params["cursor"] = cursor

        r = requests.get(
            f"{API_URL}/app.bsky.feed.getAuthorFeed",
            headers=_auth_headers(token),
            params=params,
        )

        if r.status_code == 429:
            print("Rate limit hit, sleeping 60s...")
            time.sleep(60)
            continue

        if r.status_code != 200:
            print(f"Error fetching {handle}: {r.text}")
            return [], None

        data = r.json()
        feed_items = data.get("feed", [])
        if not feed_items:
            break

        for item in feed_items:
            post_obj = item.get("post", item)
            all_posts.append(post_obj)
        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(sleep_sec)

    return all_posts, cursor
