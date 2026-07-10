"""Shared evaluation utilities for RecSys baseline models.

Metrics
-------
* Recall@k      – fraction of a user's positive test items that appear in top-k recs
* NDCG@k        – normalised discounted cumulative gain at rank k
* Coverage@k    – % of catalog items that appear in at least one user's top-k list
"""

from __future__ import annotations

import math
from typing import Dict, List, Set

import numpy as np


def _dcg(hits: List[int], k: int) -> float:
    """Compute DCG@k for a binary relevance list (hits[i] = 1 if item i is relevant)."""
    score = 0.0
    for rank, rel in enumerate(hits[:k], start=1):
        if rel:
            score += 1.0 / math.log2(rank + 1)
    return score


def recall_at_k(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 20,
) -> float:
    """Macro-averaged Recall@k across all evaluated users.

    Only users present in *both* ``recommendations`` and ``ground_truth``
    with at least one positive item are included.

    Parameters
    ----------
    recommendations : dict user_id -> ordered list of item_ids (length >= k)
    ground_truth    : dict user_id -> set of positive item_ids in test split
    k               : cut-off rank

    Returns
    -------
    float in [0, 1]
    """
    scores = []
    for user, recs in recommendations.items():
        positives = ground_truth.get(user, set())
        if not positives:
            continue
        hits = sum(1 for item in recs[:k] if item in positives)
        scores.append(hits / min(len(positives), k))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 10,
) -> float:
    """Macro-averaged NDCG@k across all evaluated users.

    Parameters
    ----------
    recommendations : dict user_id -> ordered list of item_ids (length >= k)
    ground_truth    : dict user_id -> set of positive item_ids in test split
    k               : cut-off rank

    Returns
    -------
    float in [0, 1]
    """
    scores = []
    for user, recs in recommendations.items():
        positives = ground_truth.get(user, set())
        if not positives:
            continue
        hits = [1 if item in positives else 0 for item in recs[:k]]
        dcg = _dcg(hits, k)
        # Ideal: all positives at the top
        ideal_hits = [1] * min(len(positives), k)
        idcg = _dcg(ideal_hits, k)
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(
    recommendations: Dict[str, List[str]],
    catalog: Set[str],
    k: int = 20,
) -> float:
    """Fraction of catalog items that appear in at least one user's top-k list.

    Parameters
    ----------
    recommendations : dict user_id -> ordered list of item_ids
    catalog         : complete set of item_ids in the training catalog
    k               : cut-off rank

    Returns
    -------
    float in [0, 1]
    """
    recommended = set()
    for recs in recommendations.values():
        recommended.update(recs[:k])
    return len(recommended & catalog) / len(catalog) if catalog else 0.0


def compute_all_metrics(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    catalog: Set[str],
    recall_k: int = 20,
    ndcg_k: int = 10,
) -> Dict[str, float]:
    """Convenience wrapper – returns all three baseline metrics in one call.

    Parameters
    ----------
    recommendations : dict user_id -> ordered list of item_ids
    ground_truth    : dict user_id -> set of positive item_ids
    catalog         : full set of training item_ids
    recall_k        : Recall cut-off (default 20)
    ndcg_k          : NDCG cut-off  (default 10)

    Returns
    -------
    dict with keys: recall@{k}, ndcg@{k}, coverage@{k}
    """
    return {
        f"recall@{recall_k}": recall_at_k(recommendations, ground_truth, k=recall_k),
        f"ndcg@{ndcg_k}": ndcg_at_k(recommendations, ground_truth, k=ndcg_k),
        f"coverage@{recall_k}": catalog_coverage(recommendations, catalog, k=recall_k),
    }
