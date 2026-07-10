"""Task 3 evaluation: Random vs Popularity vs MF-BPR.

This script trains / loads the baselines, fits MF-BPR, evaluates on validation
and test splits, and optionally writes a JSON summary for plot generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.metrics import compute_all_metrics
from scripts.mf_bpr import MFBPRConfig, MFBPRRecommender
from scripts.models import PopularityRecommender, RandomRecommender
try:
    from scripts.two_tower import TwoTowerConfig, TwoTowerRecommender
except Exception:  # pragma: no cover
    TwoTowerConfig = None
    TwoTowerRecommender = None


def build_history(*dfs: pd.DataFrame) -> Dict[str, Set[str]]:
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()


def build_ground_truth(df: pd.DataFrame) -> Dict[str, Set[str]]:
    positives = df[df["is_positive"]]
    return positives.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()


def eval_model(label: str, model, users: Sequence[str], history, ground_truth, catalog, k: int, ndcg_k: int):
    t0 = time.perf_counter()
    recs = model.recommend(users, history, k=k)
    rec_time = time.perf_counter() - t0
    metrics = compute_all_metrics(recs, ground_truth, catalog, recall_k=k, ndcg_k=ndcg_k)
    return {
        "model": label,
        "users": len(users),
        f"recall@{k}": metrics[f"recall@{k}"],
        f"ndcg@{ndcg_k}": metrics[f"ndcg@{ndcg_k}"],
        f"coverage@{k}": metrics[f"coverage@{k}"],
        "recommendation_time_sec": rec_time,
    }


def _sample_training_positives(train_df: pd.DataFrame, max_positives: int | None, seed: int) -> pd.DataFrame:
    if max_positives is None:
        return train_df
    positives = train_df[train_df["is_positive"]]
    if len(positives) <= max_positives:
        return train_df
    sampled_positives = positives.sample(n=max_positives, random_state=seed)
    sampled = pd.concat([train_df[~train_df["is_positive"]], sampled_positives], axis=0)
    return sampled.sort_index(kind="stable")


def evaluate(
    data_dir: Path,
    output_dir: Path,
    k: int,
    ndcg_k: int,
    mf_config: MFBPRConfig,
    two_tower_config=None,
    max_train_positives: int | None = None,
) -> Dict[str, List[Dict[str, float]]]:
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    mf_train_df = _sample_training_positives(train_df, max_train_positives, mf_config.seed)

    val_history = build_history(train_df)
    test_history = build_history(train_df, val_df)
    val_gt = build_ground_truth(val_df)
    test_gt = build_ground_truth(test_df)

    val_users = [u for u, pos in val_gt.items() if pos]
    test_users = [u for u, pos in test_gt.items() if pos]
    catalog = set(train_df["item_id"].astype(str).unique())

    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[Dict[str, float]]] = {"validation": [], "test": []}

    random_model = RandomRecommender(seed=mf_config.seed).fit(train_df, item_col="item_id")
    popularity_model = PopularityRecommender().fit(
        train_df, item_col="item_id", label_col="is_positive", use_positive_only=True
    )
    mf_model = MFBPRRecommender(config=mf_config).fit(mf_train_df, user_col="user_id", item_col="item_id", label_col="is_positive")
    two_tower_model = None
    if TwoTowerRecommender is not None and two_tower_config is not None:
        two_tower_model = TwoTowerRecommender(config=two_tower_config).fit(
            mf_train_df, user_col="user_id", item_col="item_id", label_col="is_positive"
        )
        two_tower_model.save(output_dir / "two_tower.pkl")
    mf_model.save(output_dir / "mf_bpr.pkl")
    popularity_model.save(output_dir / "popularity_recommender.pkl")
    random_model.save(output_dir / "random_recommender.pkl")

    for split_name, users, history, gt in [
        ("validation", val_users, val_history, val_gt),
        ("test", test_users, test_history, test_gt),
    ]:
        results[split_name].append(eval_model("Random", random_model, users, history, gt, catalog, k, ndcg_k))
        results[split_name].append(eval_model("Popularity", popularity_model, users, history, gt, catalog, k, ndcg_k))
        results[split_name].append(eval_model("MF-BPR", mf_model, users, history, gt, catalog, k, ndcg_k))
        if two_tower_model is not None:
            results[split_name].append(eval_model("Two-Tower", two_tower_model, users, history, gt, catalog, k, ndcg_k))

    with open(output_dir / "task3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Random, Popularity, and MF-BPR on the Steam splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=Path("."), help="Directory containing train/validation/test parquet files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--ndcg-k", type=int, default=10, dest="ndcg_k")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.0025)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-train-positives", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mf_config = MFBPRConfig(
        n_factors=args.factors,
        learning_rate=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    two_tower_config = None
    if TwoTowerConfig is not None:
        two_tower_config = TwoTowerConfig(seed=args.seed, device=args.device)
    results = evaluate(
        args.data,
        args.output_dir,
        args.k,
        args.ndcg_k,
        mf_config,
        two_tower_config=two_tower_config,
        max_train_positives=args.max_train_positives,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
