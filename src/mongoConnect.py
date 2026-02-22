import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DATABASE_NAME", "bluesky")

def get_db():
    client = MongoClient(MONGO_URL)
    return client[DB_NAME]
