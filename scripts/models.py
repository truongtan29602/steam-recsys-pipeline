"""Baseline recommender models for the Steam RecSys pipeline.

Implements two non-personalised baselines from the RecSys 1 – Foundations slides:

* **RandomRecommender** – samples items uniformly at random from the catalog
  (excluding each user's training history).  Serves as a lower-bound reference
  and is useful for sanity-checking the evaluation pipeline.

* **PopularityRecommender** – always recommends the globally most-popular items
  (by count of *positive* interactions in the training set), excluding each
  user's training history.  This baseline is surprisingly hard to beat and is a
  standard first step before personalised retrieval.

Both classes follow the same fit / recommend interface so they can be swapped
in and out of the evaluation loop without any changes to the calling code.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd


class RandomRecommender:
    """Uniform-random item recommendations, excluding each user's train history.

    This is the simplest possible baseline: it assigns equal probability to
    every item in the training catalog, regardless of popularity or any user
    signal.  Per the RecSys 1 – Foundations slides it acts as a lower bound
    whose Recall@k ≈ k / |catalog|.

    Parameters
    ----------
    seed : int
        Seed for the random-number generator.  Fix it for reproducibility.

    Usage
    -----
    >>> rec = RandomRecommender(seed=42).fit(train_df)
    >>> recs = rec.recommend(user_ids, history, k=20)
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.catalog: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, item_col: str = "parent_asin") -> "RandomRecommender":
        """Memorise the item catalog from the training interactions.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interaction table.
        item_col : str
            Column name that identifies items.

        Returns
        -------
        self
        """
        self.catalog = train_df[item_col].unique()
        return self

    def recommend(
        self,
        user_ids: List[str],
        history: Dict[str, Set[str]],
        k: int = 20,
        oversample: int = 3,
    ) -> Dict[str, List[str]]:
        """Return k uniformly-random items per user, excluding seen items.

        Uses rejection sampling rather than ``np.random.choice(replace=False)``
        on the full catalog: the catalog can be large (tens of thousands of
        games) while a user's history is tiny, so oversampling + filtering is
        far cheaper than computing a full permutation per user.

        Parameters
        ----------
        user_ids : list of str
            Users to generate recommendations for.
        history : dict mapping user_id -> set of item_ids
            Training interactions used to filter already-seen items.
        k : int
            Number of recommendations per user.
        oversample : int
            Multiplier controlling how many random candidates to draw per
            iteration before filtering.  Higher values mean fewer loop
            iterations at the cost of more memory per iteration.

        Returns
        -------
        dict mapping user_id -> list of recommended item_ids (length k)
        """
        if self.catalog is None:
            raise RuntimeError("Call fit() before recommend().")

        recs: Dict[str, List[str]] = {}
        n_catalog = len(self.catalog)

        for u in user_ids:
            seen = history.get(u, set())
            chosen: List[str] = []
            chosen_set: Set[str] = set()

            while len(chosen) < k:
                batch_size = (k - len(chosen)) * oversample
                idx = self.rng.integers(0, n_catalog, size=batch_size)
                for it in self.catalog[idx]:
                    if it not in seen and it not in chosen_set:
                        chosen.append(it)
                        chosen_set.add(it)
                        if len(chosen) == k:
                            break

            recs[u] = chosen

        return recs

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RandomRecommender":
        """Load the model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------


class PopularityRecommender:
    """Same globally-ranked-by-popularity list for every user, minus their own history.

    Popularity is defined as the **count of positive interactions** in the
    training set (not raw interaction count).  A game with many very short
    playtimes or low-score reviews should not outrank a genuinely loved game
    simply because it attracted lots of negative attention.

    As noted in the RecSys 1 – Foundations slides, this non-personalised
    baseline is often competitive with simple collaborative-filtering models
    on sparse datasets because popular items are, by definition, items that
    many users enjoy.

    Parameters
    ----------
    None – all state is created during ``fit()``.

    Usage
    -----
    >>> rec = PopularityRecommender().fit(train_df, label_col="label")
    >>> recs = rec.recommend(user_ids, history, k=20)
    """

    def __init__(self) -> None:
        self.ranked_items: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        item_col: str = "parent_asin",
        label_col: str = "label",
        use_positive_only: bool = True,
    ) -> "PopularityRecommender":
        """Rank items by (positive) interaction count in the training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interaction table.
        item_col : str
            Column name that identifies items.
        label_col : str
            Binary relevance column (1 = positive interaction).
        use_positive_only : bool
            If True (default), rank by count of *positive* interactions only.
            Set to False to rank by total interaction count instead.

        Returns
        -------
        self
        """
        counts_df = train_df[train_df[label_col] == 1] if use_positive_only else train_df
        self.ranked_items = counts_df[item_col].value_counts().index.to_numpy()
        return self

    def recommend(
        self,
        user_ids: List[str],
        history: Dict[str, Set[str]],
        k: int = 20,
        buffer: int = 200,
    ) -> Dict[str, List[str]]:
        """Return the top-k most popular items per user, excluding seen items.

        A small ``buffer`` beyond k is pre-filtered to handle the common case
        where a user has seen a handful of the very top items.  For highly
        active users who have exhausted the buffer, we fall back to scanning
        the full ranked list.

        Parameters
        ----------
        user_ids : list of str
            Users to generate recommendations for.
        history : dict mapping user_id -> set of item_ids
            Training interactions used to filter already-seen items.
        k : int
            Number of recommendations per user.
        buffer : int
            Extra items beyond k to consider before falling back to the full
            ranked list.

        Returns
        -------
        dict mapping user_id -> list of recommended item_ids (length <= k)
        """
        if self.ranked_items is None:
            raise RuntimeError("Call fit() before recommend().")

        recs: Dict[str, List[str]] = {}
        top_pool = self.ranked_items[: k + buffer]

        for u in user_ids:
            seen = history.get(u, set())
            chosen = [it for it in top_pool if it not in seen][:k]

            if len(chosen) < k:
                # Very active user exhausted the buffer — scan the full ranked list.
                chosen = [it for it in self.ranked_items if it not in seen][:k]

            recs[u] = chosen

        return recs

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PopularityRecommender":
        """Load the model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)