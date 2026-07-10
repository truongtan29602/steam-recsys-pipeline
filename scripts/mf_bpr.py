"""Matrix factorization with Bayesian Personalized Ranking (BPR).

This module implements a small, self-contained MF-BPR recommender from scratch:

* user/item ID indexing
* stochastic pairwise BPR optimization
* top-k recommendation with history filtering
* persistence helpers

The implementation is intentionally explicit so it can be inspected during the
project defense and reused by the comparison / plotting scripts.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MFBPRConfig:
    """Hyperparameters for MF-BPR training."""

    n_factors: int = 64
    learning_rate: float = 0.05
    reg: float = 0.0025
    epochs: int = 10
    num_negatives: int = 1
    batch_size: int = 2048
    seed: int = 42
    device: str = "cpu"


class MFBPRRecommender:
    """Matrix factorization recommender trained with pairwise BPR loss."""

    def __init__(self, config: MFBPRConfig | None = None) -> None:
        self.config = config or MFBPRConfig()
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_user: List[str] = []
        self.idx_to_item: List[str] = []
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.global_bias: float = 0.0

    @staticmethod
    def _build_index(values: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
        unique_values = list(dict.fromkeys(values))
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        return mapping, unique_values

    def fit(
        self,
        train_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        label_col: str = "is_positive",
    ) -> "MFBPRRecommender":
        """Train the model on positive interactions only."""

        positives = train_df[train_df[label_col]].copy()
        if positives.empty:
            raise ValueError("MF-BPR requires at least one positive interaction.")

        self.user_to_idx, self.idx_to_user = self._build_index(positives[user_col].astype(str))
        self.item_to_idx, self.idx_to_item = self._build_index(positives[item_col].astype(str))

        rng = np.random.default_rng(self.config.seed)
        n_users = len(self.idx_to_user)
        n_items = len(self.idx_to_item)
        self.global_bias = float(positives[label_col].mean())

        user_items: Dict[int, Set[int]] = {u: set() for u in range(n_users)}
        for user_id, item_id in zip(positives[user_col].astype(str), positives[item_col].astype(str)):
            user_items[self.user_to_idx[user_id]].add(self.item_to_idx[item_id])

        user_pos_pairs = [(u_idx, i_idx) for u_idx, items in user_items.items() for i_idx in items]
        if not user_pos_pairs:
            raise ValueError("MF-BPR could not build any user-item positive pairs.")

        user_factors = rng.normal(0.0, 0.01, size=(n_users, self.config.n_factors)).astype(np.float32)
        item_factors = rng.normal(0.0, 0.01, size=(n_items, self.config.n_factors)).astype(np.float32)
        user_bias = np.zeros(n_users, dtype=np.float32)
        item_bias = np.zeros(n_items, dtype=np.float32)

        all_items = np.arange(n_items, dtype=np.int64)
        pos_by_user = {u: np.fromiter(items, dtype=np.int64) for u, items in user_items.items()}

        for _ in range(self.config.epochs):
            order = rng.permutation(len(user_pos_pairs))
            for start in range(0, len(order), self.config.batch_size):
                batch_idx = order[start : start + self.config.batch_size]
                batch_users = np.array([user_pos_pairs[i][0] for i in batch_idx], dtype=np.int64)
                batch_pos = np.array([user_pos_pairs[i][1] for i in batch_idx], dtype=np.int64)
                batch_neg = np.empty_like(batch_pos)

                for j, u_idx in enumerate(batch_users):
                    forbidden = pos_by_user[u_idx]
                    while True:
                        candidate = int(rng.choice(all_items))
                        if candidate not in forbidden:
                            batch_neg[j] = candidate
                            break

                p_u = user_factors[batch_users]
                q_i = item_factors[batch_pos]
                q_j = item_factors[batch_neg]
                b_u = user_bias[batch_users]
                b_i = item_bias[batch_pos]
                b_j = item_bias[batch_neg]

                x_ui = np.sum(p_u * q_i, axis=1) + b_u + b_i
                x_uj = np.sum(p_u * q_j, axis=1) + b_u + b_j
                diff = np.clip(x_ui - x_uj, -35.0, 35.0)
                sigmoid_neg_diff = 1.0 / (1.0 + np.exp(diff))

                grad_u = (sigmoid_neg_diff[:, None] * (q_i - q_j)) + self.config.reg * p_u
                grad_i = (sigmoid_neg_diff[:, None] * p_u) + self.config.reg * q_i
                grad_j = (-sigmoid_neg_diff[:, None] * p_u) + self.config.reg * q_j
                grad_bu = sigmoid_neg_diff + self.config.reg * b_u
                grad_bi = sigmoid_neg_diff + self.config.reg * b_i
                grad_bj = -sigmoid_neg_diff + self.config.reg * b_j

                lr = self.config.learning_rate / len(batch_users)
                for idx, u_idx in enumerate(batch_users):
                    user_factors[u_idx] -= lr * grad_u[idx]
                    user_bias[u_idx] -= lr * grad_bu[idx]
                for idx, i_idx in enumerate(batch_pos):
                    item_factors[i_idx] -= lr * grad_i[idx]
                    item_bias[i_idx] -= lr * grad_bi[idx]
                for idx, j_idx in enumerate(batch_neg):
                    item_factors[j_idx] -= lr * grad_j[idx]
                    item_bias[j_idx] -= lr * grad_bj[idx]

        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_bias = user_bias
        self.item_bias = item_bias
        return self

    def _check_ready(self) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Call fit() before recommend().")

    def _score_all_items(self, user_idx: int) -> np.ndarray:
        self._check_ready()
        user_vec = self.user_factors[user_idx]
        scores = self.item_factors @ user_vec
        scores = scores + self.global_bias
        if self.user_bias is not None:
            scores = scores + self.user_bias[user_idx]
        if self.item_bias is not None:
            scores = scores + self.item_bias
        return scores

    def recommend(
        self,
        user_ids: Sequence[str],
        history: Mapping[str, Set[str]],
        k: int = 20,
    ) -> Dict[str, List[str]]:
        """Rank items for each user and exclude training-history items."""

        self._check_ready()
        recs: Dict[str, List[str]] = {}
        item_ids = np.array(self.idx_to_item)

        for user_id in user_ids:
            if user_id not in self.user_to_idx:
                recs[user_id] = []
                continue

            user_idx = self.user_to_idx[user_id]
            scores = self._score_all_items(user_idx).copy()
            seen = history.get(user_id, set())
            if seen:
                seen_idx = [self.item_to_idx[item] for item in seen if item in self.item_to_idx]
                scores[np.array(seen_idx, dtype=np.int64)] = -np.inf

            top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            recs[user_id] = item_ids[top_idx].tolist()

        return recs

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MFBPRRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
