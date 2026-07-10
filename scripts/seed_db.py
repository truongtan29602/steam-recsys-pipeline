#!/usr/bin/env python3
"""Seed PostgreSQL with Steam data from parquet files.

Usage: python scripts/seed_db.py
Reads data/processed/steam/*.parquet, populates postgres via docker compose exec.
"""

import sys
from pathlib import Path

import pandas as pd
import psycopg2

# Config
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "steam_recsys",
    "password": "steam_recsys_password",
    "dbname": "steam_recsys",
}
DATA_DIR = Path("data/processed/steam")
BATCH_SIZE = 5000


def seed():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # --- Items ---
    items = pd.read_parquet(DATA_DIR / "items.parquet")
    items["item_id"] = items["item_id"].astype(str)
    print(f"Seeding {len(items):,} items...")
    for _, row in items.iterrows():
        cur.execute(
            "INSERT INTO items (item_id, title, category, image_url) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (item_id) DO NOTHING",
            (row["item_id"], str(row.get("title", "")), str(row.get("category", "")),
             str(row.get("image_url", row.get("header_image", "")))),
        )
    conn.commit()
    print("  Items done.")

    # --- Users ---
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    users = train["user_id"].unique()
    print(f"Seeding {len(users):,} users...")
    for i, uid in enumerate(users):
        cur.execute(
            "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING",
            (str(uid),),
        )
        if (i + 1) % BATCH_SIZE == 0:
            conn.commit()
            print(f"  {i+1}/{len(users)} users...")
    conn.commit()
    print("  Users done.")

    # --- Interactions (test period only — to keep it small) ---
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"Seeding {len(test):,} test interactions...")
    for i, (_, row) in enumerate(test.iterrows()):
        cur.execute(
            "INSERT INTO interactions (user_id, item_id, event_type, playtime_forever, event_time) "
            "VALUES (%s, %s, %s, %s, %s)",
            (str(row["user_id"]), str(row["item_id"]), "play",
             float(row.get("hours", 0)), row["event_time"]),
        )
        if (i + 1) % BATCH_SIZE == 0:
            conn.commit()
            print(f"  {i+1}/{len(test)} interactions..." if (i + 1) % (BATCH_SIZE * 10) == 0 else "", end="")
    conn.commit()
    print("  Interactions done.")

    cur.close()
    conn.close()
    print("\nDatabase seeded.")


if __name__ == "__main__":
    seed()
