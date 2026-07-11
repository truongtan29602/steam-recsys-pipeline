#!/usr/bin/env python3
"""Create slide-ready data exploration assets for the Steam RecSys project.

The full interaction splits contain 5M rows, so this script keeps exact parquet
metadata for row counts and uses a deterministic sample for exploratory plots
that do not require exact full-dataset aggregation. The output is intended for
presentation slides, not model training.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "steam"
OUT_DIR = ROOT / "slides" / "data_exploration"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "data_exploration_summary.json"

SAMPLE_ROWS_PER_SPLIT = {
    "train": 350_000,
    "validation": 120_000,
    "test": 180_000,
}


def parquet_rows(path: Path) -> int:
    """Return exact row count from parquet metadata without loading the file."""
    return pq.ParquetFile(path).metadata.num_rows


def read_split_sample(split: str, n: int) -> pd.DataFrame:
    """Read selected columns and down-sample deterministically if needed."""
    path = DATA_DIR / f"{split}.parquet"
    cols = ["user_id", "item_id", "hours", "event_time", "is_positive"]
    df = pd.read_parquet(path, columns=cols)
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    df["split"] = split
    return df


def save_bar(ax, labels, values, title, ylabel, color="#4cc9f0", fmt=None):
    bars = ax.bar(labels, values, color=color, edgecolor="#0b132b", linewidth=0.8)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        text = fmt(value) if fmt else f"{value:,.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                ha="center", va="bottom", fontsize=9)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    split_paths = {s: DATA_DIR / f"{s}.parquet" for s in ["train", "validation", "test"]}
    row_counts = {split: parquet_rows(path) for split, path in split_paths.items()}

    samples = [read_split_sample(split, SAMPLE_ROWS_PER_SPLIT[split]) for split in split_paths]
    sample_df = pd.concat(samples, ignore_index=True)
    items = pd.read_parquet(DATA_DIR / "items.parquet")

    # Summary statistics for slide copy. Some values are sample-based by design
    # and are labelled as such in the generated JSON/slide prompt.
    split_summary = {}
    for split, df in sample_df.groupby("split"):
        split_summary[split] = {
            "exact_rows": int(row_counts[split]),
            "sample_rows": int(len(df)),
            "sample_unique_users": int(df["user_id"].nunique()),
            "sample_unique_items": int(df["item_id"].nunique()),
            "sample_positive_rate": round(float(df["is_positive"].mean()), 4),
            "sample_median_hours": round(float(df["hours"].median()), 2),
            "sample_p90_hours": round(float(df["hours"].quantile(0.90)), 2),
            "sample_start": str(df["event_time"].min().date()),
            "sample_end": str(df["event_time"].max().date()),
        }

    train_sample = sample_df[sample_df["split"] == "train"]
    user_counts = train_sample.groupby("user_id", sort=False).size()
    item_counts = train_sample.groupby("item_id", sort=False).size()
    top_categories = items["category"].fillna("Unknown").value_counts().head(12)

    summary = {
        "exact_total_interactions": int(sum(row_counts.values())),
        "exact_split_rows": row_counts,
        "catalog_items": int(len(items)),
        "known_metadata_items": int((items["category"].fillna("Unknown") != "Unknown").sum()),
        "sample_size": int(len(sample_df)),
        "sample_unique_users": int(sample_df["user_id"].nunique()),
        "sample_unique_items": int(sample_df["item_id"].nunique()),
        "sample_positive_rate": round(float(sample_df["is_positive"].mean()), 4),
        "split_summary": split_summary,
        "train_sample_user_interaction_quantiles": {
            str(k): round(float(v), 1)
            for k, v in user_counts.quantile([0.50, 0.75, 0.90, 0.95, 0.99]).items()
        },
        "train_sample_item_interaction_quantiles": {
            str(k): round(float(v), 1)
            for k, v in item_counts.quantile([0.50, 0.75, 0.90, 0.95, 0.99]).items()
        },
        "sample_hours_quantiles": {
            str(k): round(float(v), 2)
            for k, v in sample_df["hours"].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).items()
        },
        "top_categories": top_categories.to_dict(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Chronological split sizes.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    save_bar(
        ax,
        ["Train", "Validation", "Test"],
        [row_counts["train"], row_counts["validation"], row_counts["test"]],
        "Chronological split sizes",
        "Interactions",
        fmt=lambda v: f"{v/1_000_000:.1f}M",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_split_sizes.png", dpi=180)
    plt.close(fig)

    # 2) Positive rate by split.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates = [split_summary[s]["sample_positive_rate"] * 100 for s in ["train", "validation", "test"]]
    save_bar(
        ax,
        ["Train", "Validation", "Test"],
        rates,
        "Positive interaction rate by split (sample)",
        "% interactions with hours ≥ 1",
        color="#80ffdb",
        fmt=lambda v: f"{v:.1f}%",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_positive_rate_by_split.png", dpi=180)
    plt.close(fig)

    # 3) Playtime distribution on a log scale.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    clipped_hours = sample_df["hours"].clip(lower=0, upper=sample_df["hours"].quantile(0.995))
    ax.hist(np.log1p(clipped_hours), bins=60, color="#4cc9f0", edgecolor="white", linewidth=0.3)
    ax.set_title("Playtime is extremely skewed", fontsize=13, weight="bold")
    ax.set_xlabel("log(1 + hours played before review)")
    ax.set_ylabel("Sampled interactions")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_log_playtime_distribution.png", dpi=180)
    plt.close(fig)

    # 4) Long-tail activity: users and items in the train sample.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(user_counts.clip(upper=user_counts.quantile(0.99)), bins=50, color="#f72585")
    axes[0].set_title("Most users have few interactions", fontsize=12, weight="bold")
    axes[0].set_xlabel("Interactions per user (train sample, clipped p99)")
    axes[0].set_ylabel("Users")
    axes[1].hist(item_counts.clip(upper=item_counts.quantile(0.99)), bins=50, color="#7209b7")
    axes[1].set_title("Items follow a popularity long tail", fontsize=12, weight="bold")
    axes[1].set_xlabel("Interactions per item (train sample, clipped p99)")
    axes[1].set_ylabel("Items")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_long_tail_activity.png", dpi=180)
    plt.close(fig)

    # 5) Catalog category mix.
    fig, ax = plt.subplots(figsize=(9, 5))
    top_categories.sort_values().plot(kind="barh", ax=ax, color="#4361ee")
    ax.set_title("Catalog metadata: top Steam categories", fontsize=13, weight="bold")
    ax.set_xlabel("Catalog items")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_top_categories.png", dpi=180)
    plt.close(fig)

    print(f"Wrote summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote figures: {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()