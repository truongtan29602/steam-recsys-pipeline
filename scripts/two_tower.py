"""Two-tower retrieval model for Task 3.

The model is written to be imported safely even in environments where the
installed ``torch`` build is not usable. Importing the module does not require
PyTorch; attempting to instantiate/train the model without torch raises a
clear error.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

try:  # pragma: no cover - runtime availability depends on the environment.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class TwoTowerConfig:
    user_dim: int = 64
    item_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 1024
    epochs: int = 5
    learning_rate: float = 1e-3
    temperature: float = 0.07
    seed: int = 42
    device: str = "cpu"


class _Tower(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):  # type: ignore[override]
        return F.normalize(self.net(x), dim=-1)


class TwoTowerRecommender:
    def __init__(self, config: TwoTowerConfig | None = None) -> None:
        self.config = config or TwoTowerConfig()
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_user: List[str] = []
        self.idx_to_item: List[str] = []
        self.user_tower = None
        self.item_tower = None
        self.item_embeddings: np.ndarray | None = None
        self._item_features: np.ndarray | None = None
        self._user_features: np.ndarray | None = None

    def _require_torch(self) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is not available in this environment. Run the notebook on Colab/GPU or install a CPU-compatible torch build."
            )

    @staticmethod
    def _build_index(values: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
        unique_values = list(dict.fromkeys(values))
        return {value: idx for idx, value in enumerate(unique_values)}, unique_values

    def _build_user_features(self, train_df: pd.DataFrame) -> np.ndarray:
        grouped = train_df.groupby("user_id")
        features = []
        for user_id in self.idx_to_user:
            g = grouped.get_group(user_id) if user_id in grouped.groups else None
            if g is None:
                features.append(np.zeros(4, dtype=np.float32))
                continue
            features.append(
                np.array(
                    [
                        float(len(g)),
                        float(g["hours"].mean()),
                        float(g["products_owned"].mean()),
                        float(g["text_length"].mean()),
                    ],
                    dtype=np.float32,
                )
            )
        return np.vstack(features)

    def _build_item_features(self, train_df: pd.DataFrame) -> np.ndarray:
        grouped = train_df.groupby("item_id")
        features = []
        for item_id in self.idx_to_item:
            g = grouped.get_group(item_id) if item_id in grouped.groups else None
            if g is None:
                features.append(np.zeros(4, dtype=np.float32))
                continue
            features.append(
                np.array(
                    [
                        float(len(g)),
                        float(g["hours"].mean()),
                        float(g["text_length"].mean()),
                        float(g["is_positive"].mean()),
                    ],
                    dtype=np.float32,
                )
            )
        return np.vstack(features)

    def fit(
        self,
        train_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        label_col: str = "is_positive",
    ) -> "TwoTowerRecommender":
        self._require_torch()
        positives = train_df[train_df[label_col]].copy()
        if positives.empty:
            raise ValueError("TwoTower requires positive interactions.")

        self.user_to_idx, self.idx_to_user = self._build_index(positives[user_col].astype(str))
        self.item_to_idx, self.idx_to_item = self._build_index(positives[item_col].astype(str))
        self._user_features = self._build_user_features(train_df)
        self._item_features = self._build_item_features(train_df)

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
        self.user_tower = _Tower(self._user_features.shape[1], self.config.hidden_dim, self.config.user_dim).to(device)
        self.item_tower = _Tower(self._item_features.shape[1], self.config.hidden_dim, self.config.item_dim).to(device)

        opt = torch.optim.Adam(list(self.user_tower.parameters()) + list(self.item_tower.parameters()), lr=self.config.learning_rate)

        user_tensor = torch.tensor(self._user_features, dtype=torch.float32, device=device)
        item_tensor = torch.tensor(self._item_features, dtype=torch.float32, device=device)

        pairs = positives[[user_col, item_col]].astype(str).values.tolist()
        user_pos = {u: [] for u in self.idx_to_user}
        for u, i in pairs:
            user_pos[u].append(i)

        all_item_indices = np.arange(len(self.idx_to_item))
        rng = np.random.default_rng(self.config.seed)

        for _ in range(self.config.epochs):
            order = rng.permutation(len(pairs))
            for start in range(0, len(order), self.config.batch_size):
                idx = order[start : start + self.config.batch_size]
                batch_users = [pairs[i][0] for i in idx]
                batch_pos = [pairs[i][1] for i in idx]
                batch_neg = []
                for u in batch_users:
                    seen = set(user_pos[u])
                    while True:
                        candidate = self.idx_to_item[int(rng.choice(all_item_indices))]
                        if candidate not in seen:
                            batch_neg.append(candidate)
                            break

                u_idx = torch.tensor([self.user_to_idx[u] for u in batch_users], device=device)
                p_idx = torch.tensor([self.item_to_idx[i] for i in batch_pos], device=device)
                n_idx = torch.tensor([self.item_to_idx[i] for i in batch_neg], device=device)

                u_emb = self.user_tower(user_tensor[u_idx])
                p_emb = self.item_tower(item_tensor[p_idx])
                n_emb = self.item_tower(item_tensor[n_idx])

                pos_logits = (u_emb * p_emb).sum(dim=-1) / self.config.temperature
                all_item_emb = self.item_tower(item_tensor)
                all_logits = (u_emb @ all_item_emb.T) / self.config.temperature
                labels = p_idx
                loss = F.cross_entropy(all_logits, labels)
                loss = loss + 0.5 * ((u_emb - p_emb).pow(2).mean() + (u_emb - n_emb).pow(2).mean())

                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            self.item_embeddings = self.item_tower(item_tensor).detach().cpu().numpy()
        return self

    def _check_ready(self) -> None:
        if self.user_tower is None or self.item_embeddings is None:
            raise RuntimeError("Call fit() before recommend().")

    def recommend(
        self,
        user_ids: Sequence[str],
        history: Mapping[str, Set[str]],
        k: int = 20,
    ) -> Dict[str, List[str]]:
        self._check_ready()
        assert self._user_features is not None and self._item_features is not None
        device = next(self.user_tower.parameters()).device
        user_tensor = torch.tensor(self._user_features, dtype=torch.float32, device=device)  # type: ignore[arg-type]
        item_tensor = torch.tensor(self._item_features, dtype=torch.float32, device=device)  # type: ignore[arg-type]

        recs: Dict[str, List[str]] = {}
        item_ids = np.array(self.idx_to_item)
        with torch.no_grad():
            all_item_emb = self.item_tower(item_tensor).detach().cpu().numpy()
            for user_id in user_ids:
                if user_id not in self.user_to_idx:
                    recs[user_id] = []
                    continue
                u_idx = self.user_to_idx[user_id]
                u_emb = self.user_tower(user_tensor[[u_idx]]).detach().cpu().numpy()[0]
                scores = all_item_emb @ u_emb
                seen = history.get(user_id, set())
                if seen:
                    seen_idx = [self.item_to_idx[i] for i in seen if i in self.item_to_idx]
                    scores[np.array(seen_idx, dtype=np.int64)] = -np.inf
                top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                recs[user_id] = item_ids[top_idx].tolist()
        return recs

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TwoTowerRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
