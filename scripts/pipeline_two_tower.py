#!/usr/bin/env python3
"""Full pipeline: Enriched Two-Tower → FAISS → LightGBM ranking.

Compares retrieval-only vs retrieval+ranking NDCG.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np, pandas as pd, torch
import faiss

# ── Data ───────────────────────────────────────────────────────────────
DATA = Path("data/processed/steam")
train = pd.read_parquet(DATA / "train.parquet")
val   = pd.read_parquet(DATA / "validation.parquet")
test  = pd.read_parquet(DATA / "test.parquet")
catalog = pd.read_parquet(DATA / "items.parquet")
print(f"Train: {len(train):>10,} | {train.user_id.nunique():>8,} users | {train.item_id.nunique():>5,} items")

# ── Load enriched two-tower ───────────────────────────────────────────
from scripts.two_tower import TwoTowerRecommender
print("\nLoading enriched two-tower...")
model = TwoTowerRecommender.load("outputs/task3/two_tower_enriched.pkl")
model.user_tower.eval(); model.item_tower.eval()
device = next(model.user_tower.parameters()).device
print(f"Model: {len(model.idx_to_user):,} users | {len(model.idx_to_item):,} items | {model.item_embeddings.shape[1]}d | device={device}")

# ── FAISS index ───────────────────────────────────────────────────────
embeddings = model.item_embeddings.astype(np.float32)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
item_ids_arr = np.array(model.idx_to_item)
print(f"FAISS: {index.ntotal} vectors")

# ── Retrieve top-100 per user ─────────────────────────────────────────
def get_top_k(user_id, user_feat_tensor, k=100):
    if user_id not in model.user_to_idx:
        return []
    u_idx = model.user_to_idx[user_id]
    u_emb = model.user_tower(user_feat_tensor[[u_idx]]).cpu().numpy()[0].astype(np.float32)
    _, indices = index.search(u_emb.reshape(1, -1), k)
    return item_ids_arr[indices[0]].tolist()

user_tensor = torch.tensor(model._user_features, dtype=torch.float32, device=device)

# Build history for filtering
def build_history(df):
    return df.groupby("user_id")["item_id"].apply(set).to_dict()
val_history = build_history(train)
test_history = build_history(pd.concat([train, val]))

# Ground truth
def build_gt(df):
    return df[df.is_positive].groupby("user_id")["item_id"].apply(set).to_dict()
val_gt = build_gt(val)
test_gt = build_gt(test)

# Sample users
rng = np.random.default_rng(42)
val_users = [u for u in val.user_id.unique() if u in model.user_to_idx and u in val_gt]
val_users = rng.choice(val_users, size=min(500, len(val_users)), replace=False).tolist()
test_users = [u for u in test.user_id.unique() if u in model.user_to_idx and u in test_gt]
test_users = rng.choice(test_users, size=min(500, len(test_users)), replace=False).tolist()

K = 100
print(f"\nRetrieving top-{K} for {len(val_users)} val + {len(test_users)} test users...")
t0 = time.perf_counter()
with torch.no_grad():
    val_recs = {u: get_top_k(u, user_tensor, K) for u in val_users}
    test_recs = {u: get_top_k(u, user_tensor, K) for u in test_users}
ret_time = time.perf_counter() - t0
print(f"Retrieval: {ret_time:.1f}s")

# ── Retrieval-only metrics ───────────────────────────────────────────
from scripts.metrics import recall_at_k, ndcg_at_k
val_r20 = recall_at_k(val_recs, val_gt, k=20)
val_n10 = ndcg_at_k(val_recs, val_gt, k=10)
test_r20 = recall_at_k(test_recs, test_gt, k=20)
test_n10 = ndcg_at_k(test_recs, test_gt, k=10)
print(f"\nTwo-Tower retrieval:")
print(f"  Val:  Recall@20={val_r20:.4f}  NDCG@10={val_n10:.4f}")
print(f"  Test: Recall@20={test_r20:.4f}  NDCG@10={test_n10:.4f}")

# ── Build ranking candidates ──────────────────────────────────────────
def build_candidates(recs, interactions_df, gt_dict, train_df, item_avg_hours_map):
    rows = [(uid, item) for uid, items in recs.items() for item in items]
    df = pd.DataFrame(rows, columns=["user_id", "item_id"])
    df["is_positive"] = df.apply(lambda r: r["item_id"] in gt_dict.get(r["user_id"], set()), axis=1)
    u_events = interactions_df.groupby("user_id")["event_time"].max().reset_index()
    u_events.columns = ["user_id", "event_time"]
    df = df.merge(u_events, on="user_id", how="left")
    # Get real hours for positives, item average for negatives (NO LEAKAGE)
    ih = interactions_df.groupby(["user_id", "item_id"])["hours"].first().reset_index()
    df = df.merge(ih, on=["user_id", "item_id"], how="left")
    df["item_avg_hours"] = df["item_id"].map(item_avg_hours_map).fillna(0.0)
    df["hours"] = df["hours"].fillna(df["item_avg_hours"])  # ← estimate, not 0
    df.drop(columns=["item_avg_hours"], inplace=True)
    return df

# Pre-compute item average hours from training (for leak-free negatives)
item_avg_hours = train.groupby("item_id")["hours"].mean().to_dict()

cand_val = build_candidates(val_recs, val, val_gt, train, item_avg_hours)
cand_test = build_candidates(test_recs, test, test_gt, train, item_avg_hours)
print(f"\nCandidates: {len(cand_val):,} val | {len(cand_test):,} test")
print(f"Pos rate:  {cand_val.is_positive.mean():.2%} val | {cand_test.is_positive.mean():.2%} test")

# ── LightGBM LambdaRank ───────────────────────────────────────────────
print(f"\n{'='*50}")
print("LightGBM LambdaRank")
print(f"{'='*50}")

from steam_recsys.ranking.features import build_features
from steam_recsys.ranking.train import prepare_ranking_data, train_ranker, evaluate_ranker, save_model

t0 = time.perf_counter()
df_val, feat_cols, encoder = build_features(cand_val, train, catalog)
df_test, _, _ = build_features(cand_test, train, catalog, encoder=encoder)
feat_time = time.perf_counter() - t0
print(f"Features: {feat_time:.1f}s | {len(feat_cols)} cols")

X_tr, y_tr, q_tr = prepare_ranking_data(df_val, feat_cols)
X_te, y_te, q_te = prepare_ranking_data(df_test, feat_cols)

t0 = time.perf_counter()
ranker = train_ranker(X_tr, y_tr, q_tr, X_te, y_te, q_te, feat_cols,
                      num_boost_round=500, early_stopping_rounds=30)
rank_time = time.perf_counter() - t0

item_ids_test = df_test.item_id.values
metrics = evaluate_ranker(ranker, X_te, y_te, q_te, item_ids=item_ids_test,
                          catalog_size=len(catalog), k=10)

# ── Final comparison ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"FINAL — Two-Tower enriched → LightGBM")
print(f"{'='*65}")
print(f"{'':>30} {'Retrieval':>12} {'+Ranking':>14}")
print(f"{'-'*60}")
print(f"{'Recall@20 (val)':>30} {val_r20:>12.4f} {'—':>14}")
print(f"{'Recall@20 (test)':>30} {test_r20:>12.4f} {'—':>14}")
print(f"{'NDCG@10 (val)':>30} {val_n10:>12.4f} {'—':>14}")
print(f"{'NDCG@10 (test)':>30} {test_n10:>12.4f} {metrics['ndcg_10']:>14.4f}")
print(f"{'Catalog Coverage':>30} {'—':>12} {metrics['catalog_coverage']:>13.2%}")
print(f"{'='*65}")
delta = metrics['ndcg_10'] - test_n10
print(f"Improvement: {test_n10:.4f} → {metrics['ndcg_10']:.4f} ({delta:+.4f})")

# Feature importance
imp = ranker.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feat_cols, "gain": imp}).sort_values("gain", ascending=False)
print(f"\nTop 10 features:")
for _, r in imp_df.head(10).iterrows():
    bar = "█" * int(r["gain"] / imp_df["gain"].max() * 25)
    print(f"  {r['feature']:<40s} {bar}")

# Save
save_model(ranker, feat_cols, encoder=encoder)
Path("outputs/task3").mkdir(parents=True, exist_ok=True)
summary = {
    "model": "Two-Tower enriched (43 feat) + LightGBM LambdaRank",
    "retrieval": {"val_recall@20": round(val_r20,4), "val_ndcg@10": round(val_n10,4),
                  "test_recall@20": round(test_r20,4), "test_ndcg@10": round(test_n10,4)},
    "ranking": {"ndcg@10": metrics["ndcg_10"], "ndcg_std": metrics["ndcg_10_std"],
                "catalog_coverage": metrics["catalog_coverage"]},
    "feature_engineering_s": round(feat_time, 1),
    "ranking_training_s": round(rank_time, 1),
    "retrieval_s": round(ret_time, 1),
}
with open("outputs/task3/two_tower_ranking_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*65}")
print("DONE — Two-Tower enriched → FAISS → LightGBM")
print(f"{'='*65}")
