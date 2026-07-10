"""LightGBM LambdaRank training and evaluation.

Implements:
  - Query-grouped dataset construction
  - LambdaRank training with early stopping
  - NDCG@10 + Catalog Coverage evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def prepare_ranking_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "is_positive",
    group_col: str = "user_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X, y, query_sizes arrays for LightGBM LambdaRank.

    Query groups = all candidates for a single user.
    """
    df = df.sort_values(group_col).reset_index(drop=True)
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[label_col].fillna(0).astype(float).values
    query_sizes = df.groupby(group_col, sort=False).size().values
    return X, y, query_sizes


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    query_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    query_val: np.ndarray,
    feature_names: list[str],
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    verbose_eval: int = 50,
) -> lgb.Booster:
    """Train LightGBM LambdaRank model.

    Returns the best iteration's booster.
    """
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "boosting_type": "gbdt",
        "num_leaves": 128,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 1e-3,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=query_train,
        feature_name=feature_names,
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=query_val,
        feature_name=feature_names,
        reference=train_data,
    )

    evals_result: dict = {}

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=verbose_eval),
            lgb.record_evaluation(evals_result),
        ],
    )

    return model


def evaluate_ranker(
    model: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    query_test: np.ndarray,
    catalog_size: int,
    k: int = 10,
) -> dict:
    """Evaluate NDCG@k and Catalog Coverage.

    Returns dict with per-query NDCG and aggregate metrics.
    """
    scores = model.predict(X_test)

    ndcg_list: list[float] = []
    recommended_items: set[int] = set()

    offset = 0
    for qsize in query_test:
        if qsize == 0:
            continue
        end = offset + qsize
        group_y = y_test[offset:end]
        group_scores = scores[offset:end]

        if group_y.sum() > 0 and len(group_y) >= k:
            # Top-k indices within this group
            top_k = np.argsort(group_scores)[::-1][:k]
            recommended_items.update(top_k + offset)  # global index

            try:
                ndcg = ndcg_score([group_y], [group_scores], k=k)
                ndcg_list.append(float(ndcg))
            except ValueError:
                pass

        offset = end

    return {
        "ndcg_10": round(np.mean(ndcg_list), 4) if ndcg_list else 0.0,
        "ndcg_10_std": round(np.std(ndcg_list), 4) if ndcg_list else 0.0,
        "num_users_evaluated": len(ndcg_list),
        "catalog_coverage": round(len(recommended_items) / catalog_size, 4),
    }


def save_model(model: lgb.Booster, feature_names: list[str],
               model_dir: str = "models", artifact_dir: str = "artifacts") -> None:
    """Export LightGBM model and feature list for API serving."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    model.save_model(f"{model_dir}/ranker.txt")
    (Path(artifact_dir) / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2)
    )


def load_model(model_dir: str = "models", artifact_dir: str = "artifacts") -> tuple:
    """Load exported model and feature names for API."""
    model = lgb.Booster(model_file=f"{model_dir}/ranker.txt")
    feature_names = json.loads(
        (Path(artifact_dir) / "feature_names.json").read_text()
    )
    return model, feature_names
