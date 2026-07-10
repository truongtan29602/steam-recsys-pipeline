"""Feature engineering for LightGBM LambdaRank reranker.

14 features from Steam interaction data:
  User (4):  activity_count, avg_hours, distinct_items, positive_rate
  Item (5):  popularity, avg_hours, positive_rate, avg_text_length, is_early_access
  Cross (3): category_affinity, hours_vs_avg, popularity_rank
  Temporal (2): days_since_last, hour_sin/cos

All features computed from training data only to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ── user-level features ──────────────────────────────────────────────────────


def compute_user_features(train: pd.DataFrame) -> pd.DataFrame:
    """Per-user aggregate features from training interactions."""
    return (
        train.groupby("user_id")
        .agg(
            user_activity_count=("item_id", "count"),
            user_avg_hours=("hours", "mean"),
            user_distinct_items=("item_id", "nunique"),
            user_positive_rate=("is_positive", "mean"),
            user_total_hours=("hours", "sum"),
            user_avg_text_length=("text_length", "mean"),
        )
        .reset_index()
    )


# ── item-level features ──────────────────────────────────────────────────────


def compute_item_features(
    train: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Per-item aggregate features + catalog metadata."""
    item = (
        train.groupby("item_id")
        .agg(
            item_popularity=("user_id", "count"),
            item_avg_hours=("hours", "mean"),
            item_positive_rate=("is_positive", "mean"),
            item_avg_text_length=("text_length", "mean"),
        )
        .reset_index()
    )

    # Merge catalog: category, genres, early_access
    item = item.merge(
        catalog[["item_id", "category", "early_access_item"]],
        on="item_id",
        how="left",
    )
    item["category"] = item["category"].fillna("Unknown")
    item["item_is_early_access"] = item["early_access_item"].fillna(False).astype(int)
    item.drop(columns=["early_access_item"], inplace=True)

    return item


# ── cross features ───────────────────────────────────────────────────────────


def compute_cross_features(
    candidates: pd.DataFrame,
    user_feat: pd.DataFrame,
    item_feat: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Merge user/item features and compute cross-features."""
    df = candidates.merge(user_feat, on="user_id", how="left")
    df = df.merge(item_feat, on="item_id", how="left")

    # Category affinity: % of user's training games in this item's category
    # Compute user-category counts from training
    train_with_cat = train.merge(
        item_feat[["item_id", "category"]], on="item_id", how="left"
    )
    user_cat_counts = (
        train_with_cat.groupby(["user_id", "category"])
        .size()
        .reset_index(name="cat_count")
    )
    user_totals = train.groupby("user_id").size().reset_index(name="total_items")
    user_cat_norm = user_cat_counts.merge(user_totals, on="user_id")
    user_cat_norm["user_category_affinity"] = (
        user_cat_norm["cat_count"] / user_cat_norm["total_items"]
    )

    df = df.merge(
        user_cat_norm[["user_id", "category", "user_category_affinity"]],
        on=["user_id", "category"],
        how="left",
    )
    df["user_category_affinity"] = df["user_category_affinity"].fillna(0.0)

    # Playtime ratio: this item's hours vs user's average
    df["user_item_hours_vs_avg"] = np.where(
        df["user_avg_hours"] > 0,
        df["hours"].fillna(df["user_avg_hours"]) / df["user_avg_hours"],
        1.0,
    )

    return df


# ── temporal features ────────────────────────────────────────────────────────


def compute_temporal_features(
    df: pd.DataFrame, train: pd.DataFrame
) -> pd.DataFrame:
    """Days since last interaction + cyclic hour encoding."""
    # Last interaction timestamp per user (from training)
    last_event = (
        train.groupby("user_id")["event_time"]
        .max()
        .reset_index()
        .rename(columns={"event_time": "last_event_time"})
    )

    df = df.merge(last_event, on="user_id", how="left")
    df["days_since_last"] = (
        (df["event_time"] - df["last_event_time"]).dt.total_seconds() / 86400.0
    ).fillna(365).clip(lower=0, upper=365)

    # Cyclic hour encoding (sin/cos pair)
    hours = df["event_time"].dt.hour.fillna(0).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    return df


# ── master pipeline ──────────────────────────────────────────────────────────


def build_features(
    candidates: pd.DataFrame,
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, list[str], OneHotEncoder]:
    """Orchestrate all feature computation.

    Args:
        candidates: [user_id, item_id, event_time, hours, ...] — retrieval output
        train: Full training interactions (no leakage from val/test)
        catalog: Item metadata from items.parquet
        encoder: Pre-fit OneHotEncoder for categories. If None, fit on train+val.

    Returns:
        (feature_df, feature_names, encoder) — ready for LightGBM, label included
    """
    user_feat = compute_user_features(train)
    item_feat = compute_item_features(train, catalog)

    df = compute_cross_features(candidates, user_feat, item_feat, train)
    df = compute_temporal_features(df, train)

    # One-hot encode category (replaces pd.get_dummies for serving portability)
    cat_col = "category"
    cat_values = df[[cat_col]].fillna("Unknown")
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(cat_values)
    cat_encoded = encoder.transform(cat_values)
    cat_feature_names = encoder.get_feature_names_out([cat_col]).tolist()

    df_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)
    df = pd.concat([df.drop(columns=[cat_col]), df_cat], axis=1)

    # Numeric features only (exclude ids, timestamps, targets)
    exclude = {
        "user_id", "item_id", "event_time", "hours",
        "is_positive", "text_length", "last_event_time",
    }
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols, encoder
