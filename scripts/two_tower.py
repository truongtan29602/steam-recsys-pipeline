"""Two-tower retrieval model with catalog-enriched features.

Adds genre one-hot, price, release year, early access, developer popularity
to item features. User features get category diversity, avg_price, early_access rate.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None; nn = None; F = None


@dataclass(frozen=True)
class TwoTowerConfig:
    user_dim: int = 64
    item_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1e-3
    temperature: float = 0.07
    seed: int = 42
    device: str = "cpu"
    n_top_genres: int = 15
    n_top_tags: int = 20


class _Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
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
        self._user_features: np.ndarray | None = None
        self._item_features: np.ndarray | None = None
        self._top_genres: List[str] = []
        self._top_tags: List[str] = []
        self._popular_items: List[str] = []

    def _require_torch(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not available.")

    @staticmethod
    def _build_index(values: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
        unique_values = list(dict.fromkeys(values))
        return {value: idx for idx, value in enumerate(unique_values)}, unique_values

    def _build_user_features(self, train_df: pd.DataFrame,
                             catalog: pd.DataFrame | None = None) -> np.ndarray:
        """Build user features: interaction stats + content diversity."""
        user_feats = train_df.groupby("user_id").agg(
            feat_count=("item_id", "count"),
            feat_avg_hours=("hours", "mean"),
            feat_std_hours=("hours", "std"),
            feat_products=("products_owned", "mean"),
            feat_text_len=("text_length", "mean"),
        ).fillna(0)

        if catalog is not None:
            catalog = catalog.copy()
            if "category" not in catalog.columns:
                catalog["category"] = "Unknown"
            if "price" not in catalog.columns:
                catalog["price"] = 0.0
            if "early_access_item" not in catalog.columns:
                catalog["early_access_item"] = False
            if "genres" not in catalog.columns:
                catalog["genres"] = ""
            if "tags" not in catalog.columns:
                catalog["tags"] = ""
            if "release_date" not in catalog.columns:
                catalog["release_date"] = pd.NaT
            if "developer" not in catalog.columns:
                catalog["developer"] = "Unknown"
            # Category diversity: number of distinct genres user plays
            train_w_cat = train_df.merge(catalog[["item_id", "category", "price", "early_access_item"]],
                                         on="item_id", how="left")
            cat_div = train_w_cat.groupby("user_id")["category"].nunique().rename("feat_cat_div")
            user_feats["feat_cat_div"] = cat_div

            # Average price
            avg_price = train_w_cat.groupby("user_id")["price"].mean().rename("feat_avg_price")
            user_feats["feat_avg_price"] = avg_price.fillna(0)

            # Early access rate
            ea_rate = (train_w_cat.groupby("user_id")["early_access_item"].mean()
                       .rename("feat_ea_rate"))
            user_feats["feat_ea_rate"] = ea_rate.fillna(0)

        user_feats = user_feats.reindex(self.idx_to_user, fill_value=0)
        user_cols = [c for c in user_feats.columns if c.startswith("feat_")]
        return user_feats[user_cols].fillna(0).values.astype(np.float32)

    def _build_item_features(self, train_df: pd.DataFrame,
                             catalog: pd.DataFrame) -> np.ndarray:
        """Build item features: interaction stats + catalog metadata (genres, price, etc.)."""
        catalog = catalog.copy()
        if "category" not in catalog.columns:
            catalog["category"] = "Unknown"
        if "genres" not in catalog.columns:
            catalog["genres"] = ""
        if "tags" not in catalog.columns:
            catalog["tags"] = ""
        if "price" not in catalog.columns:
            catalog["price"] = 0.0
        if "release_date" not in catalog.columns:
            catalog["release_date"] = pd.NaT
        if "early_access_item" not in catalog.columns:
            catalog["early_access_item"] = False
        if "developer" not in catalog.columns:
            catalog["developer"] = "Unknown"
        # Interaction stats
        item_feats = train_df.groupby("item_id").agg(
            feat_popularity=("user_id", "count"),
            feat_avg_hours=("hours", "mean"),
            feat_pos_rate=("is_positive", "mean"),
            feat_text_len=("text_length", "mean"),
        ).fillna(0)

        # Merge catalog
        cat_cols = ["item_id", "category", "genres", "tags", "price",
                    "release_date", "early_access_item", "developer"]
        item_feats = item_feats.reset_index().merge(
            catalog[cat_cols], on="item_id", how="left"
        ).set_index("item_id")

        # Top genres (one-hot)
        all_genres = []
        for g in catalog["genres"].dropna():
            all_genres.extend(str(g).split("|"))
        genre_counts = pd.Series(all_genres).value_counts()
        self._top_genres = genre_counts.head(self.config.n_top_genres).index.tolist()

        for genre in self._top_genres:
            col = f"genre_{genre.replace(' ', '_').replace('&', 'and')}"
            item_feats[col] = item_feats["genres"].fillna("").apply(
                lambda x: 1.0 if genre in str(x) else 0.0
            )

        # Top tags (one-hot)
        all_tags = []
        for t in catalog["tags"].dropna():
            all_tags.extend(str(t).split("|"))
        tag_counts = pd.Series(all_tags).value_counts()
        self._top_tags = tag_counts.head(self.config.n_top_tags).index.tolist()

        for tag in self._top_tags:
            col = f"tag_{tag.replace(' ', '_').replace('&', 'and')}"
            item_feats[col] = item_feats["tags"].fillna("").apply(
                lambda x: 1.0 if tag in str(x) else 0.0
            )

        # Price: log1p normalized
        item_feats["feat_log_price"] = np.log1p(
            item_feats["price"].fillna(0).clip(lower=0)
        )

        # Release year normalized (0-1)
        item_feats["release_date"] = pd.to_datetime(item_feats["release_date"], errors="coerce")
        item_feats["feat_release_year"] = (
            item_feats["release_date"].dt.year.fillna(2015) - 2000
        ).clip(lower=0) / 30.0  # 2000-2030 range

        # Early access
        item_feats["feat_early_access"] = item_feats["early_access_item"].fillna(0).astype(float)

        # Developer popularity (how many games from this dev in catalog)
        dev_pop = catalog.groupby("developer").size().rename("dev_count")
        item_feats["feat_dev_pop"] = (item_feats["developer"]
                                      .map(dev_pop).fillna(1).clip(upper=500) / 500.0)

        # Reindex + extract feature columns
        item_feats = item_feats.reindex(self.idx_to_item)
        feat_cols = [c for c in item_feats.columns
                     if c.startswith("feat_") or c.startswith("genre_") or c.startswith("tag_")]
        result = item_feats[feat_cols].fillna(0).values.astype(np.float32)
        return result

    def fit(
        self,
        train_df: pd.DataFrame,
        catalog: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        label_col: str = "is_positive",
    ) -> "TwoTowerRecommender":
        self._require_torch()
        train_df = train_df.copy()
        train_df[user_col] = train_df[user_col].astype(str)
        train_df[item_col] = train_df[item_col].astype(str)
        positives = train_df[train_df[label_col]].copy()
        if positives.empty:
            raise ValueError("TwoTower requires positive interactions.")

        self.user_to_idx, self.idx_to_user = self._build_index(positives[user_col].astype(str))
        self.item_to_idx, self.idx_to_item = self._build_index(positives[item_col].astype(str))
        self._popular_items = positives[item_col].value_counts().index.astype(str).tolist()
        self._user_features = self._build_user_features(train_df, catalog)
        self._item_features = self._build_item_features(train_df, catalog)

        print(f"  User dim: {self._user_features.shape[1]} features")
        print(f"  Item dim: {self._item_features.shape[1]} features")

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        device = torch.device(self.config.device if torch.cuda.is_available()
                              or self.config.device == "cpu" else "cpu")
        self.user_tower = _Tower(self._user_features.shape[1], self.config.hidden_dim,
                                 self.config.user_dim).to(device)
        self.item_tower = _Tower(self._item_features.shape[1], self.config.hidden_dim,
                                 self.config.item_dim).to(device)

        opt = torch.optim.Adam(
            list(self.user_tower.parameters()) + list(self.item_tower.parameters()),
            lr=self.config.learning_rate, weight_decay=1e-5)

        user_tensor = torch.tensor(self._user_features, dtype=torch.float32, device=device)
        item_tensor = torch.tensor(self._item_features, dtype=torch.float32, device=device)

        pairs = positives[[user_col, item_col]].astype(str).values.tolist()
        user_pos = {u: [] for u in self.idx_to_user}
        for u, i in pairs:
            user_pos[u].append(i)

        all_item_indices = np.arange(len(self.idx_to_item))
        rng = np.random.default_rng(self.config.seed)

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            order = rng.permutation(len(pairs))
            n_batches = 0
            for start in range(0, len(order), self.config.batch_size):
                idx = order[start: start + self.config.batch_size]
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

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    u_emb = self.user_tower(user_tensor[u_idx])
                    p_emb = self.item_tower(item_tensor[p_idx])
                    n_emb = self.item_tower(item_tensor[n_idx])

                    # In-batch candidates: positives + sampled negatives for a much cheaper softmax.
                    cand_emb = torch.cat([p_emb, n_emb], dim=0)
                    logits = (u_emb @ cand_emb.T) / self.config.temperature
                    labels = torch.arange(len(batch_users), device=device)
                    loss = F.cross_entropy(logits, labels)
                    loss = loss + 0.1 * ((u_emb - p_emb).pow(2).sum(dim=1).mean()
                                         + (u_emb - n_emb).pow(2).sum(dim=1).mean())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs} — loss: {total_loss/n_batches:.4f}")

        with torch.no_grad():
            self.item_embeddings = self.item_tower(item_tensor).detach().cpu().numpy()
        return self

    def _check_ready(self) -> None:
        if self.user_tower is None or self.item_embeddings is None:
            raise RuntimeError("Call fit() before recommend().")

    def recommend(
        self, user_ids: Sequence[str], history: Mapping[str, Set[str]], k: int = 20
    ) -> Dict[str, List[str]]:
        self._check_ready()
        assert self._user_features is not None and self._item_features is not None
        device = next(self.user_tower.parameters()).device
        user_tensor = torch.tensor(self._user_features, dtype=torch.float32, device=device)
        item_tensor = torch.tensor(self._item_features, dtype=torch.float32, device=device)

        recs: Dict[str, List[str]] = {}
        item_ids = np.array(self.idx_to_item)
        with torch.no_grad():
            all_item_emb = self.item_tower(item_tensor).detach().cpu().numpy()
        for user_id in user_ids:
            if user_id not in self.user_to_idx:
                fallback = [it for it in self._popular_items if it not in history.get(user_id, set())][:k]
                recs[user_id] = fallback
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
