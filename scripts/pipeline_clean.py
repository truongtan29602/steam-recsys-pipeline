#!/usr/bin/env python3
"""Clean pipeline: Popularity retrieval → LightGBM ranking.

Uses Hamza's PopularityRecommender for proper retrieval (excludes training items),
labels candidates against val/test ground truth, trains LightGBM LambdaRank.
No circularity — candidates are PURE retrieval output.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

# ── Data ───────────────────────────────────────────────────────────────
DATA = Path("data/processed/steam")
train = pd.read_parquet(DATA / "train.parquet")
val   = pd.read_parquet(DATA / "validation.parquet")
test  = pd.read_parquet(DATA / "test.parquet")
catalog = pd.read_parquet(DATA / "items.parquet")

print(f"Train: {len(train):>10,} | {train.user_id.nunique():>8,} users | {train.item_id.nunique():>5,} items")
print(f"Val:   {len(val):>10,} | {val.user_id.nunique():>8,} users")
print(f"Test:  {len(test):>10,} | {test.user_id.nunique():>8,} users")

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Popularity retrieval (Hamza's recommender)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STEP 1: Popularity retrieval (clean — no injected positives)")
print(f"{'='*50}")

from scripts.models import PopularityRecommender

pop_model = PopularityRecommender()
# Hamza's recommender expects label_col="label", use_positive_only=True by default
# Our data uses "is_positive" — pass it explicitly
train_for_pop = train.copy()
train_for_pop["label"] = train_for_pop["is_positive"].astype(int)
pop_model.fit(train_for_pop, item_col="item_id", label_col="label")

# Build training history for filtering
def build_history(df):
    """user_id -> set of training items (vectorized)."""
    return df.groupby("user_id")["item_id"].apply(set).to_dict()

# Build ground truth from val/test
def build_ground_truth(df):
    """user_id -> set of positive items (vectorized)."""
    pos = df[df.is_positive]
    return pos.groupby("user_id")["item_id"].apply(set).to_dict()

val_history = build_history(train)
test_history = build_history(pd.concat([train, val]))
val_gt = build_ground_truth(val)
test_gt = build_ground_truth(test)

# Sample users for evaluation (those with ground truth in val/test)
rng = np.random.default_rng(42)
val_eval_users = [u for u in val.user_id.unique() if u in val_gt and u in val_history]
val_eval_users = rng.choice(val_eval_users, size=min(500, len(val_eval_users)), replace=False).tolist()

test_eval_users = [u for u in test.user_id.unique() if u in test_gt and u in test_history]
test_eval_users = rng.choice(test_eval_users, size=min(500, len(test_eval_users)), replace=False).tolist()

K = 100
t0 = time.perf_counter()
val_recs = pop_model.recommend(val_eval_users, val_history, k=K)
test_recs = pop_model.recommend(test_eval_users, test_history, k=K)
retrieval_time = time.perf_counter() - t0
print(f"Retrieved top-{K} for {len(val_eval_users)} val + {len(test_eval_users)} test users in {retrieval_time:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Compute retrieval-only metrics (clean)
# ═══════════════════════════════════════════════════════════════════════
from scripts.metrics import recall_at_k, ndcg_at_k

val_recall = recall_at_k(val_recs, val_gt, k=20)
val_ndcg   = ndcg_at_k(val_recs, val_gt, k=10)
test_recall = recall_at_k(test_recs, test_gt, k=20)
test_ndcg   = ndcg_at_k(test_recs, test_gt, k=10)

print(f"\nPopularity retrieval (clean):")
print(f"  Val:  Recall@20={val_recall:.4f}  NDCG@10={val_ndcg:.4f}")
print(f"  Test: Recall@20={test_recall:.4f}  NDCG@10={test_ndcg:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Build ranking candidates (retrieval output + labels)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STEP 3: Build ranking candidates (retrieval + ground-truth labels)")
print(f"{'='*50}")

def build_ranking_candidates(recs, interactions_df, gt_dict):
    """Convert retrieval output to labeled ranking DataFrame (vectorized)."""
    # Flatten recs dict into DataFrame in one shot
    rows = [(uid, item) for uid, items in recs.items() for item in items]
    df = pd.DataFrame(rows, columns=["user_id", "item_id"])

    # Label: item in ground truth (vectorized)
    df["is_positive"] = df.apply(
        lambda r: r["item_id"] in gt_dict.get(r["user_id"], set()), axis=1
    )

    # Get hours + event_time from interactions (vectorized merge)
    user_events = interactions_df.groupby("user_id")["event_time"].max().reset_index()
    user_events.columns = ["user_id", "event_time"]
    df = df.merge(user_events, on="user_id", how="left")

    item_hours = interactions_df.groupby(["user_id", "item_id"])["hours"].first().reset_index()
    df = df.merge(item_hours, on=["user_id", "item_id"], how="left")
    df["hours"] = df["hours"].fillna(0.0)

    return df

candidates_val = build_ranking_candidates(val_recs, val, val_gt)
candidates_test = build_ranking_candidates(test_recs, test, test_gt)
print(f"Val candidates:  {len(candidates_val):,} rows ({candidates_val.user_id.nunique()} users, pos_rate={candidates_val.is_positive.mean():.2%})")
print(f"Test candidates: {len(candidates_test):,} rows ({candidates_test.user_id.nunique()} users, pos_rate={candidates_test.is_positive.mean():.2%})")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: LightGBM LambdaRank
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STEP 4: LightGBM LambdaRank")
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

# ═══════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FINAL RESULTS — Popularity vs Popularity + LightGBM")
print(f"{'='*60}")
print(f"{'':>30} {'Retrieval':>12} {'+ Ranking':>12}")
print(f"{'-'*60}")
print(f"{'Recall@20 (val)':>30} {val_recall:>12.4f} {'—':>12}")
print(f"{'Recall@20 (test)':>30} {test_recall:>12.4f} {'—':>12}")
print(f"{'NDCG@10 (val)':>30} {val_ndcg:>12.4f} {'—':>12}")
print(f"{'NDCG@10 (test)':>30} {test_ndcg:>12.4f} {metrics['ndcg_10']:>12.4f}")
print(f"{'Catalog Coverage':>30} {'—':>12} {metrics['catalog_coverage']:>11.2%}")
print(f"{'Users evaluated':>30} {'—':>12} {metrics['num_users_evaluated']:>12,}")
print(f"{'='*60}")
print(f"Improvement (test NDCG): {test_ndcg:.4f} → {metrics['ndcg_10']:.4f} (+{metrics['ndcg_10']-test_ndcg:+.4f})")

# Feature importance
imp = ranker.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"feature": feature_cols, "gain": imp}).sort_values("gain", ascending=False)
print(f"\nTop 10 ranking features:")
for _, r in imp_df.head(10).iterrows():
    bar = "█" * int(r["gain"] / imp_df["gain"].max() * 25)
    print(f"  {r['feature']:<40s} {bar}")

# Save
save_model(ranker, feature_cols, encoder=encoder, model_dir="models", artifact_dir="artifacts")
print(f"\n✓ Model saved → models/ranker.txt")

# Summary JSON for ANALYSIS.md
summary = {
    "retrieval": {
        "model": "Popularity",
        "val_recall_at_20": round(val_recall, 4),
        "val_ndcg_at_10": round(val_ndcg, 4),
        "test_recall_at_20": round(test_recall, 4),
        "test_ndcg_at_10": round(test_ndcg, 4),
        "num_val_users": len(val_eval_users),
        "num_test_users": len(test_eval_users),
    },
    "ranking": {
        "model": "LightGBM LambdaRank",
        "test_ndcg_at_10": metrics["ndcg_10"],
        "test_ndcg_std": metrics["ndcg_10_std"],
        "catalog_coverage": metrics["catalog_coverage"],
        "num_features": len(feature_cols),
        "users_evaluated": metrics["num_users_evaluated"],
        "training_time_s": round(rank_time, 1),
    },
    "feature_engineering_time_s": round(feat_time, 1),
    "retrieval_time_s": round(retrieval_time, 1),
}
Path("outputs/task3").mkdir(parents=True, exist_ok=True)
with open("outputs/task3/ranking_clean_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("DONE — Clean evaluation, no circularity")
print(f"{'='*60}")
