#!/usr/bin/env python3
"""Exact mirror of notebooks/04_ranking_lightgbm.ipynb — runnable as a script.

Produces: models/ranker.txt, artifacts/, reports/figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ═══════════════════════════════════════════════════════════════════════════
# Cell 0: Setup & Imports
# ═══════════════════════════════════════════════════════════════════════════
import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 100

from steam_recsys.ranking.features import build_features
from steam_recsys.ranking.train import (
    evaluate_ranker,
    prepare_ranking_data,
    save_model,
    train_ranker,
)

DATA_DIR = Path("data/processed/steam")
Path("reports/figures").mkdir(parents=True, exist_ok=True)
print("Imports OK\n")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 1: Load Processed Data
# ═══════════════════════════════════════════════════════════════════════════
train = pd.read_parquet(DATA_DIR / "train.parquet")
val   = pd.read_parquet(DATA_DIR / "validation.parquet")
test  = pd.read_parquet(DATA_DIR / "test.parquet")
catalog = pd.read_parquet(DATA_DIR / "items.parquet")

print(f"Train:  {len(train):>10,} interactions | {train['user_id'].nunique():>8,} users | {train['item_id'].nunique():>8,} items")
print(f"Val:    {len(val):>10,} interactions | {val['user_id'].nunique():>8,} users | {val['item_id'].nunique():>8,} items")
print(f"Test:   {len(test):>10,} interactions | {test['user_id'].nunique():>8,} users | {test['item_id'].nunique():>8,} items")
print(f"Catalog: {len(catalog):>8,} items")
print(f"\nSparsity: {1 - len(train) / (train['user_id'].nunique() * train['item_id'].nunique()):.4%}")
print(f"Positive rate: {train['is_positive'].mean():.2%} (hours >= 1.0)")
print(f"Categories: {catalog['category'].nunique()}")
print(f"Top categories: {catalog['category'].value_counts().head(5).to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 2: Simulate Retrieval Candidates (placeholder for FAISS top-100)
# ═══════════════════════════════════════════════════════════════════════════
def simulate_candidates(
    interactions: pd.DataFrame,
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    n_negatives: int = 99,
    seed: int = 42,
    max_users: int = 500,
) -> pd.DataFrame:
    """Simulate retrieval candidates: 1 positive + N negatives per user.

    Sampling max_users to keep runtime reasonable.
    Replace with real FAISS top-100 from Task 3 when ready.
    """
    rng = np.random.default_rng(seed)
    all_items = set(catalog["item_id"].unique())
    all_users = interactions["user_id"].unique()
    sampled_users = rng.choice(all_users, size=min(max_users, len(all_users)), replace=False)

    rows = []
    for user_id in sampled_users:
        train_user_items = set(train[train["user_id"] == user_id]["item_id"].unique())
        user_group = interactions[interactions["user_id"] == user_id]
        positive_items = set(user_group[user_group["is_positive"]]["item_id"].unique())
        if not positive_items:
            continue

        pos_item = rng.choice(list(positive_items))
        pos_row = user_group[user_group["item_id"] == pos_item].iloc[0]
        rows.append({
            "user_id": user_id, "item_id": pos_item,
            "hours": float(pos_row["hours"]), "event_time": pos_row["event_time"],
            "is_positive": True,
        })

        neg_pool = list(all_items - train_user_items - positive_items)
        n_actual = min(n_negatives, len(neg_pool))
        if n_actual > 0:
            for neg_item in rng.choice(neg_pool, size=n_actual, replace=False):
                rows.append({
                    "user_id": user_id, "item_id": neg_item,
                    "hours": 0.0, "event_time": pos_row["event_time"],
                    "is_positive": False,
                })

    return pd.DataFrame(rows)


print("\nGenerating val candidates...")
candidates_val = simulate_candidates(val, train, catalog)
print(f"Val candidates: {len(candidates_val):,} rows ({candidates_val['user_id'].nunique():,} users)")
print(f"  Positive: {candidates_val['is_positive'].sum():,} | Negative: {(~candidates_val['is_positive']).sum():,}")

print("\nGenerating test candidates...")
candidates_test = simulate_candidates(test, train, catalog)
print(f"Test candidates: {len(candidates_test):,} rows ({candidates_test['user_id'].nunique():,} users)")
print(f"  Positive: {candidates_test['is_positive'].sum():,} | Negative: {(~candidates_test['is_positive']).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 3: Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
df_val, feature_cols, encoder = build_features(candidates_val, train, catalog)
df_test, _, _ = build_features(candidates_test, train, catalog, encoder=encoder)
elapsed = time.perf_counter() - t0
print(f"\nFeature engineering: {elapsed:.1f}s")
print(f"{len(feature_cols)} features (after one-hot encoding category):")
for i, col in enumerate(feature_cols):
    print(f"  {i+1:2d}. {col}")
print(f"\nVal feature matrix:  {df_val.shape}")
print(f"Test feature matrix: {df_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 4: Feature Correlation
# ═══════════════════════════════════════════════════════════════════════════
corr = df_val[feature_cols].corr()
high_corr = [
    (feature_cols[i], feature_cols[j], corr.iloc[i, j])
    for i in range(len(feature_cols))
    for j in range(i + 1, len(feature_cols))
    if abs(corr.iloc[i, j]) > 0.9
]
if high_corr:
    print(f"\n⚠ {len(high_corr)} pairs with correlation > 0.9:")
    for a, b, v in high_corr:
        print(f"  {a} ↔ {b}: {v:.3f}")
else:
    print("\n✓ No features with correlation > 0.9")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 5: Train LightGBM LambdaRank
# ═══════════════════════════════════════════════════════════════════════════
X_train, y_train, q_train = prepare_ranking_data(df_val, feature_cols)
X_test, y_test, q_test = prepare_ranking_data(df_test, feature_cols)

print(f"\nTraining set: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]:,} rows")
print(f"Queries:      {len(q_train):,} (train) | {len(q_test):,} (test)")
print(f"Avg group:    {q_train.mean():.0f} (train) | {q_test.mean():.0f} (test)")

t0 = time.perf_counter()
model = train_ranker(
    X_train, y_train, q_train,
    X_test, y_test, q_test,
    feature_names=feature_cols,
    num_boost_round=500,
    early_stopping_rounds=30,
)
train_time = time.perf_counter() - t0
print(f"\nTraining: {train_time:.1f}s | Best iter: {model.best_iteration}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 6: Evaluate
# ═══════════════════════════════════════════════════════════════════════════
item_ids_test = df_test["item_id"].values
catalog_size = len(catalog)
metrics = evaluate_ranker(model, X_test, y_test, q_test,
                          item_ids=item_ids_test, catalog_size=catalog_size, k=10)

print(f"\n{'=' * 60}")
print("LightGBM LambdaRank — Test Results (REAL STEAM DATA)")
print(f"{'=' * 60}")
print(f"NDCG@10:           {metrics['ndcg_10']:.4f} ± {metrics['ndcg_10_std']:.4f}")
print(f"Catalog Coverage:  {metrics['catalog_coverage']:.2%}")
print(f"Users evaluated:   {metrics['num_users_evaluated']:,}")
print(f"Training time:     {train_time:.1f}s")
print(f"Catalog size:      {catalog_size:,}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 7: Feature Importance
# ═══════════════════════════════════════════════════════════════════════════
importance = model.feature_importance(importance_type="gain")
importance_df = (
    pd.DataFrame({"feature": feature_cols, "gain": importance})
    .sort_values("gain", ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df["feature"][::-1], importance_df["gain"][::-1])
ax.set_xlabel("Gain (feature importance)")
ax.set_title("Top 20 Features — LightGBM LambdaRank (Steam Data)")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance_real.png", dpi=100)
plt.close()

print(f"\nTop 15 features by gain:")
for _, row in importance_df.head(15).iterrows():
    bar = "█" * int(row["gain"] / importance_df["gain"].max() * 30)
    print(f"  {row['feature']:<40s} {bar}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 8: Latency Breakdown
# ═══════════════════════════════════════════════════════════════════════════
sample = df_test.iloc[:100]
X_sample = sample[feature_cols].values.astype(np.float32)
times = []
for _ in range(200):
    t0 = time.perf_counter()
    scores = model.predict(X_sample)
    top10 = np.argsort(scores)[::-1][:10]
    times.append(time.perf_counter() - t0)

avg_ms = np.mean(times) * 1000
p99_ms = np.percentile(times, 99) * 1000
print(f"\nLatency (100 candidates → LightGBM predict + top-10):")
print(f"  Avg: {avg_ms:.2f} ms")
print(f"  P99: {p99_ms:.2f} ms")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 9: Export Model for API
# ═══════════════════════════════════════════════════════════════════════════
save_model(model, feature_cols, encoder=encoder)
print(f"\nExported: models/ranker.txt + artifacts/feature_names.json + category_encoder.pkl")
for p in sorted(Path("models").glob("ranker*")):
    print(f"  {p} ({p.stat().st_size / 1024:.1f} KB)")
for p in sorted(Path("artifacts").glob("*")):
    if p.suffix in (".json", ".pkl"):
        print(f"  {p} ({p.stat().st_size / 1024:.1f} KB)")

# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PIPELINE COMPLETE — Real Steam Data")
print(f"{'=' * 60}")
