"""Standalone runner for the Random recommender baseline.

Evaluation protocol
-------------------
- Fit   on train
- Val   evaluation: history = train
- Test  evaluation: history = train + validation

Usage
-----
    python scripts/random_rec.py [--data data/processed] [--k 20] [--seed 42]

For the full comparison (both models + Markdown report) use:
    python scripts/evaluate_baselines.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.metrics import compute_all_metrics
from scripts.models import RandomRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RandomRecommender on the Steam dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",   type=Path, default=Path("data/processed"))
    p.add_argument("--k",      type=int,  default=20,  help="Recall / Coverage cut-off")
    p.add_argument("--ndcg-k", type=int,  default=10,  dest="ndcg_k")
    p.add_argument("--seed",   type=int,  default=42)
    return p.parse_args()


def _build_history(*dfs: pd.DataFrame) -> dict:
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby("user_id")["item_id"].apply(set).to_dict()


def _build_gt(df: pd.DataFrame) -> dict:
    return df[df["is_positive"]].groupby("user_id")["item_id"].apply(set).to_dict()


def main() -> None:
    args = parse_args()

    print("Loading data …")
    train_df = pd.read_parquet(args.data / "train.parquet")
    val_df   = pd.read_parquet(args.data / "validation.parquet")
    test_df  = pd.read_parquet(args.data / "test.parquet")

    val_history  = _build_history(train_df)
    test_history = _build_history(train_df, val_df)
    val_gt       = _build_gt(val_df)
    test_gt      = _build_gt(test_df)

    val_users  = [u for u, pos in val_gt.items()  if pos]
    test_users = [u for u, pos in test_gt.items() if pos]
    catalog    = set(train_df["item_id"].unique())

    print(f"Catalog: {len(catalog):,}  |  Val users: {len(val_users):,}  |  Test users: {len(test_users):,}")

    print(f"\nFitting RandomRecommender (seed={args.seed}) …")
    t0 = time.perf_counter()
    model = RandomRecommender(seed=args.seed).fit(train_df, item_col="item_id")
    print(f"  fit: {time.perf_counter()-t0:.2f}s")

    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    model.save(model_path / "random_recommender.pkl")
    print(f"  saved to {model_path / 'random_recommender.pkl'}")

    for split_name, users, history, gt in [
        ("Validation", val_users,  val_history,  val_gt),
        ("Test",       test_users, test_history, test_gt),
    ]:
        print(f"\nGenerating recs for {len(users):,} {split_name} users …")
        t1 = time.perf_counter()
        recs = model.recommend(users, history, k=args.k)
        print(f"  rec: {time.perf_counter()-t1:.2f}s")

        metrics = compute_all_metrics(recs, gt, catalog, recall_k=args.k, ndcg_k=args.ndcg_k)
        print(f"\n── RandomRecommender [{split_name}] ────────────────────────")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")

    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
