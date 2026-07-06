"""Data preparation utilities for the Steam recommender project.

The UCSD Steam files are gzip-compressed, line-delimited Python literals rather
than strict JSON. These helpers stream the files row by row and keep the logic
shared between the command-line script and the narrated notebook.
"""

from __future__ import annotations

import ast
import gzip
import math
from pathlib import Path
from typing import Iterator

import pandas as pd

RAW_DIR = Path("data/raw/steam")
PROCESSED_DIR = Path("data/processed/steam")
FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")

REVIEWS_PATH = RAW_DIR / "steam_reviews.json.gz"
GAMES_PATH = RAW_DIR / "steam_games.json.gz"

POSITIVE_HOURS_THRESHOLD = 1.0
MAX_INTERACTIONS = 5_000_000
VALIDATION_FRACTION_OF_TRAIN = 0.10
TEST_FRACTION = 0.20


def iter_literal_gz(path: Path) -> Iterator[dict]:
    """Yield dictionaries from a gzipped line-delimited Python-literal file."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Could not parse {path} line {line_number}") from exc


def load_games(path: Path = GAMES_PATH) -> pd.DataFrame:
    """Load and clean Steam game metadata."""
    rows = []
    for record in iter_literal_gz(path):
        item_id = record.get("id")
        if item_id is None:
            continue
        genres = record.get("genres") or []
        tags = record.get("tags") or []
        rows.append(
            {
                "item_id": str(item_id),
                "title": record.get("title") or record.get("app_name") or f"Steam app {item_id}",
                "category": genres[0] if genres else (tags[0] if tags else "Unknown"),
                "genres": "|".join(map(str, genres)) if isinstance(genres, list) else "",
                "tags": "|".join(map(str, tags)) if isinstance(tags, list) else "",
                "release_date": record.get("release_date"),
                "developer": record.get("developer"),
                "publisher": record.get("publisher"),
                "price": record.get("price"),
                "early_access_item": bool(record.get("early_access", False)),
                "store_url": record.get("url"),
            }
        )

    games = pd.DataFrame(rows).drop_duplicates("item_id", keep="last")
    games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")
    return games


def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    """Load and clean Steam review interactions."""
    rows = []
    for record in iter_literal_gz(path):
        user_id = record.get("username")
        item_id = record.get("product_id")
        event_time = record.get("date")
        if not user_id or not item_id or not event_time:
            continue

        hours = pd.to_numeric(record.get("hours"), errors="coerce")
        rows.append(
            {
                "user_id": str(user_id),
                "item_id": str(item_id),
                "hours": float(hours) if pd.notna(hours) else math.nan,
                "event_time": event_time,
                "text_length": len(str(record.get("text") or "")),
                "early_access_review": bool(record.get("early_access", False)),
                "products_owned": pd.to_numeric(record.get("products"), errors="coerce"),
                "found_funny": pd.to_numeric(record.get("found_funny", 0), errors="coerce"),
                "received_for_free": "free" in str(record.get("compensation", "")).lower(),
            }
        )

    reviews = pd.DataFrame(rows)
    reviews["event_time"] = pd.to_datetime(reviews["event_time"], errors="coerce")
    reviews["hours"] = pd.to_numeric(reviews["hours"], errors="coerce")
    reviews = reviews.dropna(subset=["user_id", "item_id", "event_time", "hours"])
    reviews = reviews[reviews["hours"] >= 0].copy()

    # If a user reviewed the same item more than once, keep the latest signal.
    reviews = (
        reviews.sort_values(["user_id", "item_id", "event_time"])
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
    reviews["is_positive"] = reviews["hours"] >= POSITIVE_HOURS_THRESHOLD
    return reviews


def subsample_most_recent(interactions: pd.DataFrame, max_rows: int = MAX_INTERACTIONS) -> pd.DataFrame:
    """Keep the most recent interactions if the dataset exceeds the assignment cap."""
    ordered = interactions.sort_values("event_time").reset_index(drop=True)
    if len(ordered) <= max_rows:
        return ordered
    return ordered.tail(max_rows).reset_index(drop=True)


def time_based_split(
    interactions: pd.DataFrame,
    validation_fraction_of_train: float = VALIDATION_FRACTION_OF_TRAIN,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create mandatory chronological train/validation/test splits."""
    ordered = interactions.sort_values("event_time").reset_index(drop=True)
    test_start_idx = int(len(ordered) * (1 - test_fraction))
    train_val = ordered.iloc[:test_start_idx].copy()
    test = ordered.iloc[test_start_idx:].copy()

    val_start_idx = int(len(train_val) * (1 - validation_fraction_of_train))
    train = train_val.iloc[:val_start_idx].copy()
    validation = train_val.iloc[val_start_idx:].copy()

    boundaries = {
        "train_start": train["event_time"].min(),
        "train_end": train["event_time"].max(),
        "validation_start": validation["event_time"].min(),
        "validation_end": validation["event_time"].max(),
        "test_start": test["event_time"].min(),
        "test_end": test["event_time"].max(),
    }
    return train, validation, test, boundaries


def build_catalog(interactions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Join observed items to metadata and create serving-friendly item rows."""
    observed = pd.DataFrame({"item_id": sorted(interactions["item_id"].unique())})
    catalog = observed.merge(games, on="item_id", how="left")
    catalog["title"] = catalog["title"].fillna("Steam app " + catalog["item_id"].astype(str))
    catalog["category"] = catalog["category"].fillna("Unknown")
    catalog["image_url"] = (
        "https://cdn.akamai.steamstatic.com/steam/apps/" + catalog["item_id"].astype(str) + "/header.jpg"
    )
    return catalog


def summarize(interactions: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """Return key Task 1 metrics used by the notebook/report."""
    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()
    possible = n_users * n_items
    observed = len(interactions)
    sparsity = 1 - observed / possible if possible else float("nan")
    positive_rate = interactions["is_positive"].mean()
    return {
        "interactions": int(observed),
        "users": int(n_users),
        "items": int(n_items),
        "catalog_items": int(len(catalog)),
        "items_with_known_metadata": int((catalog["category"] != "Unknown").sum()),
        "sparsity": float(sparsity),
        "positive_threshold_hours": POSITIVE_HOURS_THRESHOLD,
        "positive_interactions": int(interactions["is_positive"].sum()),
        "positive_rate": float(positive_rate),
        "start_date": interactions["event_time"].min(),
        "end_date": interactions["event_time"].max(),
    }


def save_outputs(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, catalog: pd.DataFrame) -> None:
    """Persist split data for later baselines, retrieval, ranking, API loading."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep_interaction_cols = [
        "user_id",
        "item_id",
        "hours",
        "event_time",
        "is_positive",
        "text_length",
        "early_access_review",
        "products_owned",
        "found_funny",
        "received_for_free",
    ]
    train[keep_interaction_cols].to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    validation[keep_interaction_cols].to_parquet(PROCESSED_DIR / "validation.parquet", index=False)
    test[keep_interaction_cols].to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    catalog.to_parquet(PROCESSED_DIR / "items.parquet", index=False)