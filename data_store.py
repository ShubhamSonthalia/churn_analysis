"""
Data persistence layer.

The original app required a live MongoDB instance just to start up.
This version keeps MongoDB support (handy if you want a real database
behind the "Customer Lookup" tab) but makes it fully optional: if
MONGO_URI isn't set, or the database isn't reachable, the app falls
back to the bundled CSV files automatically so it always runs.
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "").strip()
_collection = None
_connection_error = None

if MONGO_URI:
    try:
        from pymongo import MongoClient

        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        _client.admin.command("ping")
        _collection = _client["churn_analysis"]["customers"]
    except Exception as exc:  # pragma: no cover - depends on external service
        _connection_error = str(exc)
        _collection = None


def mongo_enabled() -> bool:
    """Whether a live MongoDB connection is available."""
    return _collection is not None


def storage_mode_label() -> str:
    if mongo_enabled():
        return "MongoDB (live)"
    if MONGO_URI:
        return "Local CSV (MongoDB configured but unreachable)"
    return "Local CSV (no MONGO_URI set)"


def sync_customers(df: pd.DataFrame) -> None:
    """Best-effort upsert of scored customer rows into MongoDB, if enabled."""
    if not mongo_enabled():
        return
    try:
        records = df.to_dict("records")
        for record in records:
            key = record.get("customerid", record.get("CustomerID"))
            if key is None:
                continue
            _collection.update_one({"CustomerID": key}, {"$set": record}, upsert=True)
    except Exception as exc:  # pragma: no cover - depends on external service
        print(f"[data_store] MongoDB sync skipped: {exc}")


def fetch_customer(customer_id) -> dict | None:
    """Look up a single customer document from MongoDB, if enabled."""
    if not mongo_enabled():
        return None
    try:
        return _collection.find_one({"CustomerID": customer_id})
    except Exception as exc:  # pragma: no cover - depends on external service
        print(f"[data_store] MongoDB lookup failed: {exc}")
        return None
