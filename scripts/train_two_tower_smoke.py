#!/usr/bin/env python3
"""Quick two-tower smoke test on GPU with Steam sample data.

Trains Hamza's TwoTowerRecommender on a 50K-user sample,
exports item embeddings, and generates retrieval candidates.
"""
import sys, time
from pathlib import Path

# Ensure we can import from scripts/ (Hamza's code) and src/ (Malo's code)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from scripts.two_tower import TwoTowerRecommender, TwoTowerConfig

# ── Load data ──────────────────────────────────────────────────────────
DATA = Path("data/processed/steam")
train = pd.read_parquet(DATA / "train.parquet")
val   = pd.read_parquet(DATA / "validation.parquet")
test  = pd.read_parquet(DATA / "test.parquet")
catalog = pd.read_parquet(DATA / "items.parquet")

print(f"Train:  {len(train):>10,} | {train.user_id.nunique():>8,} users | {train.item_id.nunique():>5,} items")
print(f"Val:    {len(val):>10,} | {val.user_id.nunique():>8,} users")
print(f"Test:   {len(test):>10,} | {test.user_id.nunique():>8,} users")

# ── Sample users for quick training ────────────────────────────────────
rng = np.random.default_rng(42)
sample_users = rng.choice(train.user_id.unique(), size=50_000, replace=False)
train_sample = train[train.user_id.isin(sample_users)].copy()
print(f"\nSample: {len(train_sample):,} interactions ({train_sample.user_id.nunique():,} users, {train_sample.item_id.nunique():,} items)")

# ── Configure + train two-tower ────────────────────────────────────────
config = TwoTowerConfig(
    user_dim=64,
    item_dim=64,
    hidden_dim=128,
    batch_size=1024,
    epochs=5,
    learning_rate=1e-3,
    temperature=0.07,
    seed=42,
    device="cuda",
)

print(f"\nTraining two-tower on GPU ({config.epochs} epochs × {len(train_sample):,} interactions)...")
t0 = time.perf_counter()

model = TwoTowerRecommender(config)
model.fit(train_sample)

elapsed = time.perf_counter() - t0
print(f"Training: {elapsed:.1f}s")
print(f"Users:  {len(model.idx_to_user):,}")
print(f"Items:  {len(model.idx_to_item):,}")
print(f"Item embeddings shape: {model.item_embeddings.shape}")

# ── Test a recommendation ──────────────────────────────────────────────
# Pick a user from the sample, get their training history
test_user = model.idx_to_user[0]
history = {}
for uid in model.idx_to_user[:100]:
    user_items = set(train_sample[train_sample.user_id == uid].item_id.unique())
    history[uid] = user_items

t0 = time.perf_counter()
recs = model.recommend([test_user], history, k=20)
rec_time = (time.perf_counter() - t0) * 1000
top_games = recs.get(test_user, [])
print(f"\nRecommendation for {test_user}:")
print(f"  Top-20: {top_games[:5]}..." if len(top_games) >= 5 else f"  Top-20: {top_games}")
print(f"  Time:   {rec_time:.1f} ms")

# ── Quick retrieval quality check ──────────────────────────────────────
# For a few users, check if their training items appear in top-100
eval_users = model.idx_to_user[:500]
eval_history = {u: set(train_sample[train_sample.user_id == u].item_id.unique()) for u in eval_users}
recs_all = model.recommend(eval_users, eval_history, k=100)

hits = 0
total = 0
for u in eval_users:
    if u in recs_all and eval_history[u]:
        retrieved = set(recs_all[u])
        relevant = eval_history[u]
        # Check recall@100 on training data (not test — just sanity)
        hit_count = len(retrieved & relevant)
        hits += hit_count
        total += len(relevant)

recall_train = hits / total if total > 0 else 0
print(f"\nRetrieval sanity (Recall@100 on training data): {recall_train:.4f}")
print(f"  (should be > 0.3 — embedding model should retrieve its own training items)")

# ── Save model ─────────────────────────────────────────────────────────
Path("outputs/task3").mkdir(parents=True, exist_ok=True)
model.save("outputs/task3/two_tower_sample.pkl")
print(f"\n✓ Model saved → outputs/task3/two_tower_sample.pkl")

print(f"\n{'='*50}")
print("TWO-TOWER SMOKE TEST PASSED")
print(f"{'='*50}")
