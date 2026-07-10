"""Steam RecSys FastAPI backend — loads models at startup, serves recommendations."""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# --- Path setup ---
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from steam_recsys.ranking.features import build_features  # noqa: E402
from scripts.two_tower import TwoTowerRecommender  # noqa: E402
from scripts.models import PopularityRecommender  # noqa: E402

# --- Config ---
DATA_DIR = ROOT / "data" / "processed" / "steam"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
TT_MODEL_DIR = ROOT / "outputs" / "task3"

# ---------------------------------------------------------------------------
# Models (loaded once at startup)
# ---------------------------------------------------------------------------
two_tower = None          # TwoTowerRecommender
pop_model = None          # PopularityRecommender
ranker = None             # LightGBM booster
faiss_index = None        # FAISS IndexFlatIP
item_ids_arr = None       # np.array of item_ids (aligned with FAISS)
feature_cols = None       # LightGBM feature names
encoder = None            # OneHotEncoder
catalog_df = None         # items catalog DataFrame
item_avg_hours_map = None # item_id → avg_hours
user_tensor = None        # pre-loaded user features tensor
device = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Steam RecSys API",
    description="Two-tower retrieval → LightGBM ranking → top-10 recommendations",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GameItem(BaseModel):
    item_id: str
    title: str
    category: Optional[str] = None
    image_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    section: str
    model: str
    items: List[Dict]
    latency_ms: float


class UserInfo(BaseModel):
    user_id: str
    history_count: int


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    """Load all models into memory once."""
    global two_tower, pop_model, ranker, faiss_index, item_ids_arr
    global feature_cols, encoder, catalog_df, item_avg_hours_map, user_tensor, device

    t0 = time.perf_counter()
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")

    # --- Two-Tower ---
    tt_path = TT_MODEL_DIR / "two_tower_enriched.pkl"
    if tt_path.exists():
        two_tower = TwoTowerRecommender.load(str(tt_path))
        two_tower.user_tower.eval()
        two_tower.item_tower.eval()
        device = next(two_tower.user_tower.parameters()).device
        user_tensor = torch.tensor(two_tower._user_features, dtype=torch.float32, device=device)

        # FAISS index
        emb = two_tower.item_embeddings.astype(np.float32)
        faiss_index = faiss.IndexFlatIP(emb.shape[1])
        faiss_index.add(emb)
        item_ids_arr = np.array(two_tower.idx_to_item)
        print(f"Two-Tower loaded: {len(two_tower.idx_to_user):,} users, {len(two_tower.idx_to_item):,} items")
    else:
        print(f"WARNING: Two-tower model not found at {tt_path}")

    # --- Popularity ---
    pop_model = PopularityRecommender()
    train_for_pop = train_df.copy()
    train_for_pop["label"] = train_for_pop["is_positive"].astype(int)
    pop_model.fit(train_for_pop, item_col="item_id", label_col="label")
    print("Popularity loaded.")

    # --- LightGBM ---
    ranker_path = MODEL_DIR / "ranker.txt"
    if ranker_path.exists():
        ranker = lgb.Booster(model_file=str(ranker_path))
        feat_path = ARTIFACT_DIR / "feature_names.json"
        feature_cols = json.loads(feat_path.read_text()) if feat_path.exists() else []
        enc_path = ARTIFACT_DIR / "category_encoder.pkl"
        encoder = pickle.loads(enc_path.read_bytes()) if enc_path.exists() else None
        print(f"LightGBM loaded: {ranker.num_trees()} trees, {len(feature_cols)} features")
    else:
        print("WARNING: LightGBM model not found — ranking disabled")

    # --- Catalog ---
    catalog_df = pd.read_parquet(DATA_DIR / "items.parquet")
    catalog_df["item_id"] = catalog_df["item_id"].astype(str)
    item_avg_hours_map = train_df.groupby("item_id")["hours"].mean().to_dict()
    print(f"Catalog loaded: {len(catalog_df):,} items")

    print(f"Startup: {time.perf_counter() - t0:.1f}s — ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_popular_history() -> Dict[str, Set[str]]:
    """Build user history dict for popularity filtering."""
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    hist = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    return hist


def build_item_response(item_id: str) -> Dict:
    """Build a single item dict from catalog."""
    row = catalog_df[catalog_df["item_id"] == item_id]
    if row.empty:
        return {"item_id": item_id, "title": f"Game {item_id}", "category": "Unknown"}
    r = row.iloc[0]
    return {
        "item_id": str(r["item_id"]),
        "title": str(r.get("title", f"Game {item_id}")),
        "category": str(r.get("category", "Unknown")),
        "image_url": str(r.get("image_url", "")),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "two_tower": two_tower is not None,
        "ranker": ranker is not None,
        "faiss_items": len(item_ids_arr) if item_ids_arr is not None else 0,
    }


@app.get("/users", response_model=List[UserInfo])
def list_users(limit: int = Query(100, le=500)):
    """List test-period users for the frontend dropdown."""
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    users = test_df.groupby("user_id").size().reset_index(name="history_count")
    users = users.sort_values("history_count", ascending=False).head(limit)
    return [UserInfo(user_id=row["user_id"], history_count=int(row["history_count"]))
            for _, row in users.iterrows()]


@app.get("/recommendations/popular", response_model=RecommendationResponse)
def recommend_popular():
    """Guest homepage — top-20 popular items."""
    t0 = time.perf_counter()
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    pop_items = train_df.groupby("item_id").size().sort_values(ascending=False).head(20).index.tolist()
    items = [build_item_response(i) for i in pop_items]
    elapsed = (time.perf_counter() - t0) * 1000
    return RecommendationResponse(
        section="🏠 Homepage — Trending",
        model="Popularity",
        items=items,
        latency_ms=round(elapsed, 2),
    )


@app.get("/recommendations/personalized/{user_id}", response_model=RecommendationResponse)
def recommend_personalized(user_id: str, k: int = Query(10, le=50)):
    """Logged-in homepage — Two-tower retrieval → LightGBM ranking."""
    if two_tower is None:
        raise HTTPException(503, "Two-tower model not loaded")

    t0 = time.perf_counter()
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")

    # --- Step 1: Two-tower retrieval ---
    if user_id not in two_tower.user_to_idx:
        # Cold user → fallback to popularity
        pop_items = train_df.groupby("item_id").size().sort_values(ascending=False).head(k).index.tolist()
        items = [build_item_response(i) for i in pop_items]
        elapsed = (time.perf_counter() - t0) * 1000
        return RecommendationResponse(
            section="🏠 Homepage — For You (cold user → popularity)",
            model="Popularity (fallback)",
            items=items,
            latency_ms=round(elapsed, 2),
        )

    u_idx = two_tower.user_to_idx[user_id]
    with torch.no_grad():
        u_emb = two_tower.user_tower(user_tensor[[u_idx]]).cpu().numpy()[0].astype(np.float32)
    _, top_indices = faiss_index.search(u_emb.reshape(1, -1), 100)
    candidate_ids = item_ids_arr[top_indices[0]].tolist()

    # --- Step 2: LightGBM ranking ---
    if ranker is not None and feature_cols:
        # Build feature rows for candidates
        user_history = train_df[train_df["user_id"] == user_id]
        rows = []
        for item_id in candidate_ids:
            is_pos = item_id in set(user_history[user_history["is_positive"]]["item_id"].unique())
            hours_val = float(user_history[user_history["item_id"] == item_id]["hours"].iloc[0]) \
                if item_id in set(user_history["item_id"].unique()) else item_avg_hours_map.get(item_id, 0.0)
            event_t = user_history["event_time"].max() if len(user_history) > 0 else pd.Timestamp.now()
            rows.append({"user_id": user_id, "item_id": item_id, "hours": hours_val,
                         "event_time": event_t, "is_positive": is_pos})
        cand_df = pd.DataFrame(rows)

        try:
            feat_df, _, _ = build_features(cand_df, train_df, catalog_df, encoder=encoder)
            X = feat_df[feature_cols].values.astype(np.float32)
            scores = ranker.predict(X)
            top10 = np.argsort(-scores)[:k]
            ranked_ids = [candidate_ids[i] for i in top10]
        except Exception:
            ranked_ids = candidate_ids[:k]
    else:
        ranked_ids = candidate_ids[:k]

    items = [build_item_response(i) for i in ranked_ids]
    elapsed = (time.perf_counter() - t0) * 1000
    return RecommendationResponse(
        section="🏠 Homepage — For You",
        model="Two-Tower + LightGBM LambdaRank",
        items=items,
        latency_ms=round(elapsed, 2),
    )


@app.get("/items/{item_id}/similar", response_model=RecommendationResponse)
def recommend_similar(item_id: str, k: int = Query(10, le=50)):
    """Item page — cosine similarity on two-tower item embeddings."""
    if two_tower is None or faiss_index is None:
        raise HTTPException(503, "Two-tower model not loaded")

    t0 = time.perf_counter()
    if item_id not in two_tower.item_to_idx:
        raise HTTPException(404, f"Item {item_id} not found in model catalog")

    item_emb = two_tower.item_embeddings[two_tower.item_to_idx[item_id]].astype(np.float32)
    _, top_idx = faiss_index.search(item_emb.reshape(1, -1), k + 1)
    similar = [item_ids_arr[i] for i in top_idx[0] if item_ids_arr[i] != item_id][:k]
    items = [build_item_response(i) for i in similar]
    elapsed = (time.perf_counter() - t0) * 1000
    return RecommendationResponse(
        section=f"🎮 Similar to {item_id}",
        model="Two-Tower item embeddings (cosine)",
        items=items,
        latency_ms=round(elapsed, 2),
    )


@app.get("/users/{user_id}/history", response_model=RecommendationResponse)
def user_history(user_id: str, k: int = Query(10, le=50)):
    """Get a user's play history — for 'Because you liked X' rows."""
    t0 = time.perf_counter()
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    user_data = test_df[test_df["user_id"] == user_id]
    if user_data.empty:
        raise HTTPException(404, f"User {user_id} not found in test set")

    # Get most-played games
    top_games = (user_data.groupby("item_id")["hours"].sum()
                 .sort_values(ascending=False).head(k).index.tolist())
    items = [build_item_response(i) for i in top_games]
    elapsed = (time.perf_counter() - t0) * 1000
    return RecommendationResponse(
        section=f"🎮 {user_id}'s History",
        model="Play history (test period)",
        items=items,
        latency_ms=round(elapsed, 2),
    )


@app.get("/recommendations/because/{user_id}/{item_id}", response_model=RecommendationResponse)
def recommend_because(user_id: str, item_id: str, k: int = Query(10, le=50)):
    """'Because you liked X' — similar items to one from user history."""
    # Just re-use the similar items endpoint logic
    return recommend_similar(item_id, k)
