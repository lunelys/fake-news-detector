import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

BLUESKY_USERNAME = os.getenv("BSKY_IDENTIFIER")
BLUESKY_PASSWORD = os.getenv("BSKY_PASSWORD")

API_URL = "https://bsky.social/xrpc"

def load_token():
    """Authentifie l'utilisateur et retourne un token JWT Bluesky."""
    payload = {
        "identifier": BLUESKY_USERNAME,
        "password": BLUESKY_PASSWORD
    }

    r = requests.post(f"{API_URL}/com.atproto.server.createSession", json=payload)

    if r.status_code != 200:
        print("Erreur d'authentification Bluesky :", r.text)
        return None

    return r.json().get("accessJwt")

def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}

def fetch_feed_paginated(token, feed_uri, limit_per_page=50, max_pages=10, sleep_sec=0.3):
    """
    Fetch posts from a feed using pagination with rate-limit safety.
    Stops early if no new posts are returned on a page.

    Args:
        token: Bluesky JWT token
        feed_uri: feed identifier
        limit_per_page: number of posts per API call
        max_pages: maximum number of pages to fetch
        sleep_sec: seconds to sleep between API calls
    Returns:
        List of posts
    """
    all_posts = []
    cursor = None

    for page in range(max_pages):
        params = {"feed": feed_uri, "limit": limit_per_page}
        if cursor:
            params["cursor"] = cursor

        r = requests.get(f"{API_URL}/app.bsky.feed.getFeed",
                         headers=_auth_headers(token),
                         params=params)

        if r.status_code == 429:
            print("Rate limit hit, sleeping 60s...")
            time.sleep(60)
            continue

        if r.status_code != 200:
            print(f"Erreur feed {feed_uri} (page {page+1}):", r.text)
            break

        data = r.json()
        feed_items = data.get("feed", [])

        if not feed_items:
            print(f"No new posts found on page {page+1}, stopping early.")
            break

        all_posts.extend(feed_items)

        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(sleep_sec)  # small pause to avoid rate-limit

    return all_posts

# Note : How did I get those feed ? With f"{API_URL}/app.bsky.feed.getSuggestedFeeds" (examples of feed that I might like)
# And then I chose some with big likes number. Only the "What's Hot" feed is an official one
# I chose Verified News to get a "golden feed", and the Ukrainan feed to maybe get some false news/less verified news to compare

# Feed-specific functions
def get_hot_feed(token, limit=50):
    return fetch_feed_paginated(
        token,
        "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot",
        limit_per_page=limit
    )

def get_ukrainian_feed(token, limit=50):
    return fetch_feed_paginated(
        token,
        "at://did:plc:dvgliotey33vix3wlltybgkd/app.bsky.feed.generator/ukrainian-view",
        limit_per_page=limit
    )

def get_science_feed(token, limit=50):
    return fetch_feed_paginated(
        token,
        "at://did:plc:jfhpnnst6flqway4eaeqzj2a/app.bsky.feed.generator/for-science",
        limit_per_page=limit
    )

def get_verified_news_feed(token, limit=50):
    return fetch_feed_paginated(
        token,
        "at://did:plc:kkf4naxqmweop7dv4l2iqqf5/app.bsky.feed.generator/verified-news",
        limit_per_page=limit
    )