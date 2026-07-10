#!/usr/bin/env python3
"""Full pipeline: two-tower (GPU) → FAISS → retrieval candidates → LightGBM ranking.

Trains Hamza's TwoTowerRecommender on the full 3.6M training set,
builds FAISS index, generates top-100 retrieval candidates,
then re-trains the LightGBM LambdaRank reranker with real retrieval results.

Saves: outputs/task3/two_tower_full.pkl, models/ranker_full.txt
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np, pandas as pd
import lightgbm as lgb

# ── Data ───────────────────────────────────────────────────────────────
DATA = Path("data/processed/steam")
train = pd.read_parquet(DATA / "train.parquet")
val   = pd.read_parquet(DATA / "validation.parquet")
test  = pd.read_parquet(DATA / "test.parquet")
catalog = pd.read_parquet(DATA / "items.parquet")

print(f"Train:  {len(train):>10,} interactions | {train.user_id.nunique():>8,} users | {train.item_id.nunique():>5,} items")
print(f"Val:    {len(val):>10,} interactions | {val.user_id.nunique():>8,} users")
print(f"Test:   {len(test):>10,} interactions | {test.user_id.nunique():>8,} users")

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Full two-tower training on GPU
# ═══════════════════════════════════════════════════════════════════════
from scripts.two_tower import TwoTowerRecommender, TwoTowerConfig

config = TwoTowerConfig(
    user_dim=64, item_dim=64, hidden_dim=128,
    batch_size=2048, epochs=5, learning_rate=1e-3,
    temperature=0.07, seed=42, device="cuda",
)

print(f"\n{'='*50}")
print("STEP 1: Full two-tower training (GPU)")
print(f"{'='*50}")
print(f"Config: {config.epochs} epochs × {len(train):,} interactions, batch={config.batch_size}")

t0 = time.perf_counter()
model = TwoTowerRecommender(config)
model.fit(train)
tt_time = time.perf_counter() - t0

print(f"Training: {tt_time:.0f}s ({tt_time/60:.1f} min)")
print(f"Users:   {len(model.idx_to_user):,}")
print(f"Items:   {len(model.idx_to_item):,}")
print(f"Embeds:  {model.item_embeddings.shape}")

# Save
Path("outputs/task3").mkdir(parents=True, exist_ok=True)
model.save("outputs/task3/two_tower_full.pkl")
print(f"✓ Saved → outputs/task3/two_tower_full.pkl")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Build FAISS index
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STEP 2: FAISS index")
print(f"{'='*50}")

try:
    import faiss
    embeddings = model.item_embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine (embeddings are L2-normalized)
    index.add(embeddings)
    print(f"FAISS IndexFlatIP: {index.ntotal} vectors × {dim}d")
except ImportError:
    print("⚠ faiss not installed, using numpy fallback")
    # NumPy fallback for exact search
    embeddings = model.item_embeddings.astype(np.float32)
    index = None  # will use dot product manually

item_ids_arr = np.array(model.idx_to_item)

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Generate retrieval candidates for val/test users
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STEP 3: Retrieval candidates (top-100 per user)")
print(f"{'='*50}")

def get_top_k(user_id, k=100):
    """Retrieve top-k items for a user via FAISS or numpy dot product."""
    if user_id not in model.user_to_idx:
        return []
    u_idx = model.user_to_idx[user_id]
    u_emb = model.user_tower(
        torch.tensor(model._user_features[[u_idx]], dtype=torch.float32,
                     device=next(model.user_tower.parameters()).device)
    ).detach().cpu().numpy()[0].astype(np.float32)

    if index is not None:
        scores, indices = index.search(u_emb.reshape(1, -1), k)
        return item_ids_arr[indices[0]].tolist()
    else:
        scores = embeddings @ u_emb
        top = np.argpartition(-scores, min(k, len(scores)-1))[:k]
        top = top[np.argsort(-scores[top])]
        return item_ids_arr[top].tolist()

import torch

def generate_candidates_popularity(interactions, train_df, catalog, max_users=500, k=100):
    """Generate top-k popularity candidates + ground truth labels."""
    rng = np.random.default_rng(42)
    users = interactions.user_id.unique()
    sampled = rng.choice(users, size=min(max_users, len(users)), replace=False)

    # Top-k most popular items from training
    pop_items = train_df.groupby("item_id").size().sort_values(ascending=False).head(k).index.tolist()

    rows = []
    for uid in sampled:
        user_interactions = interactions[interactions.user_id == uid]
        pos_items = set(user_interactions[user_interactions.is_positive].item_id.unique())
        event_t = user_interactions.event_time.max()

        # Add positive items (ground truth) that may NOT be in popularity top-k
        for item in pos_items:
            rows.append({"user_id": uid, "item_id": item,
                         "hours": float(user_interactions[user_interactions.item_id == item].hours.iloc[0]),
                         "event_time": event_t, "is_positive": True})

        # Add popularity items as negatives (if not already positive)
        for item in pop_items:
            if item not in pos_items:
                rows.append({"user_id": uid, "item_id": item,
                             "hours": 0.0, "event_time": event_t, "is_positive": False})

    return pd.DataFrame(rows)

t0 = time.perf_counter()
candidates_val = generate_candidates_popularity(val, train, catalog, max_users=500)
candidates_test = generate_candidates_popularity(test, train, catalog, max_users=500)
cand_time = time.perf_counter() - t0
print(f"Val candidates:  {len(candidates_val):,} rows ({candidates_val.user_id.nunique():,} users)")
print(f"Test candidates: {len(candidates_test):,} rows ({candidates_test.user_id.nunique():,} users)")
print(f"Generation time: {cand_time:.1f}s")
print(f"Pos rate (val):  {candidates_val.is_positive.mean():.2%}")
print(f"Pos rate (test): {candidates_test.is_positive.mean():.2%}")

# Retrieval-only metrics (popularity baseline)
from scripts.metrics import recall_at_k, ndcg_at_k
pop_recs_val = {}
for uid in candidates_val.user_id.unique():
    pop_recs_val[uid] = candidates_val[candidates_val.user_id == uid].item_id.tolist()
pop_gt_val = {}
for uid in candidates_val.user_id.unique():
    pop_gt_val[uid] = set(candidates_val[(candidates_val.user_id == uid) & candidates_val.is_positive].item_id.unique())

pop_recall = recall_at_k(pop_recs_val, pop_gt_val, k=20)
pop_ndcg = ndcg_at_k(pop_recs_val, pop_gt_val, k=10)
print(f"\nPopularity-only (retrieval baseline):")
print(f"  Recall@20: {pop_recall:.4f}")
print(f"  NDCG@10:   {pop_ndcg:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: LightGBM Ranking with real retrieval candidates
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STEP 4: LightGBM LambdaRank with two-tower candidates")
print(f"{'='*50}")

from steam_recsys.ranking.features import build_features
from steam_recsys.ranking.train import prepare_ranking_data, train_ranker, evaluate_ranker, save_model

t0 = time.perf_counter()
df_val, feature_cols, encoder = build_features(candidates_val, train, catalog)
df_test, _, _ = build_features(candidates_test, train, catalog, encoder=encoder)
feat_time = time.perf_counter() - t0
print(f"Feature engineering: {feat_time:.1f}s | {len(feature_cols)} features")

X_train, y_train, q_train = prepare_ranking_data(df_val, feature_cols)
X_test, y_test, q_test = prepare_ranking_data(df_test, feature_cols)

t0 = time.perf_counter()
ranker = train_ranker(X_train, y_train, q_train, X_test, y_test, q_test,
                      feature_cols, num_boost_round=500, early_stopping_rounds=30)
rank_time = time.perf_counter() - t0

item_ids_test = df_test.item_id.values
metrics = evaluate_ranker(ranker, X_test, y_test, q_test,
                          item_ids=item_ids_test, catalog_size=len(catalog), k=10)

print(f"\n{'='*60}")
print("FINAL RESULTS — Two-Tower + LightGBM")
print(f"{'='*60}")
print(f"Two-tower training: {tt_time:.0f}s ({tt_time/60:.1f} min)")
print(f"Feature engineering: {feat_time:.1f}s")
print(f"LightGBM training:  {rank_time:.1f}s (best iter: {ranker.best_iteration})")
print(f"{'='*60}")
print(f"NDCG@10 (popularity-only):  {pop_ndcg:.4f}")
print(f"NDCG@10 (popularity+rank):  {metrics['ndcg_10']:.4f} ± {metrics['ndcg_10_std']:.4f}")
print(f"Catalog Coverage (retrieval):  — (same candidates)")
print(f"Catalog Coverage (re-ranked):  {metrics['catalog_coverage']:.2%}")
print(f"Users evaluated:             {metrics['num_users_evaluated']:,}")
print(f"{'='*60}")

# Feature importance
imp = ranker.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feature_cols, "gain": imp}).sort_values("gain", ascending=False)
print(f"\nTop 10 ranking features:")
for _, r in imp_df.head(10).iterrows():
    bar = "█" * int(r["gain"] / imp_df["gain"].max() * 25)
    print(f"  {r['feature']:<40s} {bar}")

# Save
save_model(ranker, feature_cols, encoder=encoder, model_dir="models", artifact_dir="artifacts")
print(f"\n✓ Model saved → models/ranker.txt + artifacts/")

# Write summary for ANALYSIS.md
summary = {
    "two_tower": {
        "users": len(model.idx_to_user),
        "items": len(model.idx_to_item),
        "embedding_dim": 64,
        "training_time_s": round(tt_time, 1),
        "config": config.__dict__,
    },
    "retrieval_only": {"recall_at_20": round(pop_recall, 4), "ndcg_at_10": round(pop_ndcg, 4)},
    "retrieval_plus_ranking": {
        "ndcg_10": metrics["ndcg_10"],
        "ndcg_10_std": metrics["ndcg_10_std"],
        "catalog_coverage": metrics["catalog_coverage"],
        "users_evaluated": metrics["num_users_evaluated"],
        "training_time_s": round(rank_time, 1),
    },
    "feature_engineering_time_s": round(feat_time, 1),
}
Path("outputs/task3").mkdir(parents=True, exist_ok=True)
with open("outputs/task3/ranking_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("FULL PIPELINE COMPLETE")
print(f"{'='*60}")
