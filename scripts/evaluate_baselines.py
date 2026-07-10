"""Evaluate both baseline recommenders from saved models.

Usage
-----
    python scripts/evaluate_baselines.py [--data data/processed] [--k 20] [--ndcg-k 10]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.metrics import compute_all_metrics
from scripts.models import PopularityRecommender, RandomRecommender

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_history(*dfs: pd.DataFrame) -> Dict[str, Set[str]]:
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby("user_id")["item_id"].apply(set).to_dict()


def build_ground_truth(df: pd.DataFrame) -> Dict[str, Set[str]]:
    positives = df[df["is_positive"]]
    return positives.groupby("user_id")["item_id"].apply(set).to_dict()


def eval_model(model, eval_users, history, ground_truth, catalog_set, k, ndcg_k, label: str):
    t0 = time.perf_counter()
    recs = model.recommend(eval_users, history, k=k)
    rec_time = time.perf_counter() - t0
    metrics = compute_all_metrics(recs, ground_truth, catalog_set, recall_k=k, ndcg_k=ndcg_k)
    print(
        f"      [{label}] "
        f"Recall@{k}: {metrics[f'recall@{k}']:.4f}  |  "
        f"NDCG@{ndcg_k}: {metrics[f'ndcg@{ndcg_k}']:.4f}  |  "
        f"Coverage@{k}: {metrics[f'coverage@{k}']:.4f}  "
        f"({rec_time:.1f}s)"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(data_dir: Path, k: int, ndcg_k: int) -> None:
    print("=" * 65)
    print("  Steam RecSys – Baseline Evaluation (val / test)")
    print("=" * 65)

    print("\n[1/5] Loading data …")
    t0 = time.perf_counter()
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df   = pd.read_parquet(data_dir / "validation.parquet")
    test_df  = pd.read_parquet(data_dir / "test.parquet")
    print(
        f"      Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

    print("\n[2/5] Building history / ground-truth dictionaries …")
    t0 = time.perf_counter()

    val_history  = build_history(train_df)
    test_history = build_history(train_df, val_df)
    val_gt  = build_ground_truth(val_df)
    test_gt = build_ground_truth(test_df)

    val_users  = [u for u, pos in val_gt.items()  if pos]
    test_users = [u for u, pos in test_gt.items() if pos]
    catalog_set: Set[str] = set(train_df["item_id"].unique())

    print(
        f"      Val eval users: {len(val_users):,} | "
        f"Test eval users: {len(test_users):,} | "
        f"Catalog: {len(catalog_set):,}  ({time.perf_counter()-t0:.1f}s)"
    )

    print("\n[3/5] Loading models …")
    try:
        rr = RandomRecommender.load("models/random_recommender.pkl")
        print("      RandomRecommender loaded")
    except FileNotFoundError:
        print("      RandomRecommender NOT FOUND. Run popular_rec.py and random_rec.py first.")
        return

    try:
        pr = PopularityRecommender.load("models/popularity_recommender.pkl")
        print("      PopularityRecommender loaded")
    except FileNotFoundError:
        print("      PopularityRecommender NOT FOUND. Run popular_rec.py and random_rec.py first.")
        return

    print(f"\n[4/5] Validation evaluation ({len(val_users):,} users) …")
    eval_model(rr, val_users,  val_history, val_gt,  catalog_set, k, ndcg_k, "Random")
    eval_model(pr, val_users,  val_history, val_gt,  catalog_set, k, ndcg_k, "Popularity")

    print(f"\n[5/5] Test evaluation ({len(test_users):,} users) …")
    eval_model(rr, test_users, test_history, test_gt, catalog_set, k, ndcg_k, "Random")
    eval_model(pr, test_users, test_history, test_gt, catalog_set, k, ndcg_k, "Popularity")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Random + Popularity on Steam (val/test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    type=Path, default=DATA_DIR,    help="Directory with parquet splits")
    parser.add_argument("--k",       type=int,  default=20,          help="Recall / Coverage cut-off")
    parser.add_argument("--ndcg-k",  type=int,  default=10, dest="ndcg_k", help="NDCG cut-off")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(data_dir=args.data, k=args.k, ndcg_k=args.ndcg_k)
    print("\nDone.")


if __name__ == "__main__":
    main()
