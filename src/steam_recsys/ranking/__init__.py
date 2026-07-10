"""Ranking module: feature engineering + LightGBM LambdaRank reranker.

All features are computed from training data only to avoid data leakage.
The pipeline: raw interactions → 14 features → LightGBM LambdaRank → re-ranked top-10.
"""

from .features import build_features, compute_user_features, compute_item_features
from .train import train_ranker, evaluate_ranker, prepare_ranking_data

__all__ = [
    "build_features",
    "compute_user_features",
    "compute_item_features",
    "train_ranker",
    "evaluate_ranker",
    "prepare_ranking_data",
]
