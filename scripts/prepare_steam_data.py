"""Run Task 1 Part 1 Steam data preparation from the command line."""

from __future__ import annotations

import json

import pandas as pd

from src.steam_recsys.data_prep import (
    FIGURES_DIR,
    GAMES_PATH,
    REVIEWS_PATH,
    TABLES_DIR,
    build_catalog,
    load_games,
    load_reviews,
    save_outputs,
    subsample_most_recent,
    summarize,
    time_based_split,
)


def _json_ready(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def main() -> None:
    if not REVIEWS_PATH.exists() or not GAMES_PATH.exists():
        raise FileNotFoundError(
            "Missing raw Steam files. Place steam_reviews.json.gz and steam_games.json.gz in data/raw/steam/."
        )

    print("Loading Steam game metadata...")
    games = load_games()
    print(f"Loaded {len(games):,} game metadata rows")

    print("Loading Steam reviews/interactions. This can take a few minutes for the full file...")
    interactions = load_reviews()
    raw_after_cleaning = len(interactions)
    interactions = subsample_most_recent(interactions)
    print(f"Prepared {len(interactions):,} interactions ({raw_after_cleaning:,} before assignment cap)")

    catalog = build_catalog(interactions, games)
    train, validation, test, boundaries = time_based_split(interactions)
    summary = summarize(interactions, catalog)

    save_outputs(train, validation, test, catalog)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train), "start": train.event_time.min(), "end": train.event_time.max()},
            {
                "split": "validation",
                "rows": len(validation),
                "start": validation.event_time.min(),
                "end": validation.event_time.max(),
            },
            {"split": "test", "rows": len(test), "start": test.event_time.min(), "end": test.event_time.max()},
        ]
    )
    split_summary.to_csv(TABLES_DIR / "task1_split_summary.csv", index=False)

    serializable_summary = {key: _json_ready(value) for key, value in summary.items()}
    serializable_summary["raw_interactions_after_cleaning"] = raw_after_cleaning
    serializable_summary["subsampling_strategy"] = (
        "No cap applied" if raw_after_cleaning <= len(interactions) else "Kept the most recent 5M interactions"
    )
    serializable_summary["split_boundaries"] = {key: _json_ready(value) for key, value in boundaries.items()}
    (TABLES_DIR / "task1_data_summary.json").write_text(json.dumps(serializable_summary, indent=2), encoding="utf-8")

    print("\nTask 1 Part 1 summary")
    print(json.dumps(serializable_summary, indent=2))
    print("\nSaved outputs to data/processed/steam/ and reports/tables/.")


if __name__ == "__main__":
    main()