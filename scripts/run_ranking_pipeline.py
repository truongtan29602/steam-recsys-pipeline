#!/usr/bin/env python3
"""End-to-end execution of the ranking pipeline on synthetic Steam-like data.

Mirrors all cells from notebooks/04_ranking_lightgbm.ipynb.
Run with: source .venv/bin/activate && python scripts/run_ranking_pipeline.py
"""
from __future__ import annotations

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

from steam_recsys.ranking.features import build_features
from steam_recsys.ranking.train import (
    evaluate_ranker,
    prepare_ranking_data,
    save_model,
    train_ranker,
)

Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Cell 1: Generate synthetic data (same shape as real Steam)
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)
n_users, n_items, n_train = 500, 1_000, 200_000
users = [f"u{i}" for i in range(n_users)]
items = [f"g{i}" for i in range(n_items)]
categories = ["Action", "RPG", "Strategy", "Sports", "Simulation", "Indie"]

train = pd.DataFrame({
    "user_id": np.random.choice(users, n_train),
    "item_id": np.random.choice(items, n_train),
    "hours": np.random.exponential(5, n_train),
    "is_positive": np.random.binomial(1, 0.6, n_train).astype(bool),
    "text_length": np.random.poisson(200, n_train),
    "event_time": pd.date_range("2018-01-01", periods=n_train, freq="15min"),
    "found_funny": np.zeros(n_train),
    "early_access_review": np.zeros(n_train, dtype=bool),
    "products_owned": np.random.randint(1, 200, n_train).astype(float),
    "received_for_free": np.zeros(n_train, dtype=bool),
})

catalog = pd.DataFrame({
    "item_id": items,
    "category": np.random.choice(categories, n_items),
    "early_access_item": np.random.binomial(1, 0.1, n_items),
})

sparsity = 1 - len(train) / (n_users * n_items)
print(f"Train:  {len(train):>10,} interactions | {n_users:>5,} users | {n_items:>5,} items")
print(f"Sparsity: {sparsity:.4%}")
print(f"Positive rate: {train['is_positive'].mean():.2%}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 2: Long-tail analysis
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
item_counts = train.groupby("item_id").size().sort_values(ascending=False)
user_counts = train.groupby("user_id").size().sort_values(ascending=False)
axes[0].loglog(range(1, len(item_counts) + 1), item_counts.values)
axes[0].set_title("Item Popularity (log-log)")
axes[0].set_xlabel("Item rank")
axes[1].loglog(range(1, len(user_counts) + 1), user_counts.values)
axes[1].set_title("User Activity (log-log)")
axes[1].set_xlabel("User rank")
plt.tight_layout()
plt.savefig("reports/figures/long_tail.png", dpi=100)
plt.close()
print("Long-tail plot → reports/figures/long_tail.png")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 3: Simulate retrieval candidates
# ═══════════════════════════════════════════════════════════════════════════
n_val_users = 200
val_users = np.random.choice(users, n_val_users, replace=False)
candidates = []
for uid in val_users:
    user_train = set(train[train["user_id"] == uid]["item_id"].unique())
    pos = np.random.choice(list(user_train)) if user_train else np.random.choice(items)
    candidates.append({
        "user_id": uid, "item_id": pos, "hours": 10.0,
        "event_time": pd.Timestamp("2021-06-01"), "is_positive": True,
    })
    neg_pool = list(set(items) - user_train - {pos})
    for n in np.random.choice(neg_pool, min(99, len(neg_pool)), replace=False):
        candidates.append({
            "user_id": uid, "item_id": n, "hours": 2.0,
            "event_time": pd.Timestamp("2021-06-01"), "is_positive": False,
        })

df_cand = pd.DataFrame(candidates)
print(f"\nVal candidates: {len(df_cand):,} rows, {df_cand['user_id'].nunique():,} users")
print(f"  Positive: {df_cand['is_positive'].sum():,}  |  Negative: {(~df_cand['is_positive']).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 4: Feature engineering
# ═══════════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
df_val, feature_cols, encoder = build_features(df_cand, train, catalog)
elapsed = time.perf_counter() - t0
print(f"\nFeature engineering: {elapsed:.1f}s")
print(f"{len(feature_cols)} features:")
for i, c in enumerate(feature_cols):
    print(f"  {i+1:2d}. {c}")
print(f"Feature matrix: {df_val.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 5: Correlation check
# ═══════════════════════════════════════════════════════════════════════════
corr = df_val[feature_cols].corr()
high_corr = []
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        if abs(corr.iloc[i, j]) > 0.9:
            high_corr.append((feature_cols[i], feature_cols[j], corr.iloc[i, j]))
if high_corr:
    print(f"\n⚠ {len(high_corr)} pairs with correlation > 0.9:")
    for a, b, v in high_corr:
        print(f"  {a} ↔ {b}: {v:.3f}")
else:
    print("\n✓ No features with correlation > 0.9")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 6: Train LightGBM LambdaRank
# ═══════════════════════════════════════════════════════════════════════════
X, y, q = prepare_ranking_data(df_val, feature_cols)
item_ids_all = df_val["item_id"].values  # for catalog coverage
split_q = int(len(q) * 0.8)
split_idx = q[:split_q].sum()
X_tr, y_tr, q_tr = X[:split_idx], y[:split_idx], q[:split_q]
X_va, y_va, q_va = X[split_idx:], y[split_idx:], q[split_q:]
item_ids_va = item_ids_all[split_idx:]

print(f"\nTraining:   {X_tr.shape[0]:,} rows | {len(q_tr):,} queries")
print(f"Validation: {X_va.shape[0]:,} rows | {len(q_va):,} queries")

t0 = time.perf_counter()
model = train_ranker(
    X_tr, y_tr, q_tr, X_va, y_va, q_va, feature_cols,
    num_boost_round=300, early_stopping_rounds=20, verbose_eval=30,
)
train_time = time.perf_counter() - t0
print(f"\nTraining: {train_time:.1f}s | Best iter: {model.best_iteration}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 7: Evaluate
# ═══════════════════════════════════════════════════════════════════════════
metrics = evaluate_ranker(model, X_va, y_va, q_va,
                          item_ids=item_ids_va,
                          catalog_size=n_items, k=10)
print(f"\n{'=' * 50}")
print("LightGBM LambdaRank — Results")
print(f"{'=' * 50}")
print(f"NDCG@10:           {metrics['ndcg_10']:.4f} ± {metrics['ndcg_10_std']:.4f}")
print(f"Catalog Coverage:  {metrics['catalog_coverage']:.2%}")
print(f"Users evaluated:   {metrics['num_users_evaluated']:,}")
print(f"Training time:     {train_time:.1f}s")
print(f"Catalog size:      {n_items:,}")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 8: Feature importance
# ═══════════════════════════════════════════════════════════════════════════
importance = model.feature_importance(importance_type="gain")
imp_df = (
    pd.DataFrame({"feature": feature_cols, "gain": importance})
    .sort_values("gain", ascending=False)
)
print(f"\nTop 10 features by gain:")
max_gain = imp_df["gain"].max()
for _, row in imp_df.head(10).iterrows():
    bar = "█" * int(row["gain"] / max_gain * 30)
    print(f"  {row['feature']:<40s} {bar}")

fig, ax = plt.subplots(figsize=(10, 5))
top15 = imp_df.head(15)
ax.barh(top15["feature"][::-1], top15["gain"][::-1])
ax.set_xlabel("Gain")
ax.set_title("Top 15 Features — LightGBM LambdaRank")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=100)
plt.close()
print("Feature importance plot → reports/figures/feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 9: Latency benchmark
# ═══════════════════════════════════════════════════════════════════════════
sample = df_val.iloc[:100]
X_sample = sample[feature_cols].fillna(0).values.astype(np.float32)
times = []
for _ in range(200):
    t0 = time.perf_counter()
    model.predict(X_sample)
    times.append(time.perf_counter() - t0)
avg_ms = np.mean(times) * 1000
p99_ms = np.percentile(times, 99) * 1000
print(f"\nLatency (100 candidates): avg={avg_ms:.2f}ms  p99={p99_ms:.2f}ms")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 10: Export model
# ═══════════════════════════════════════════════════════════════════════════
save_model(model, feature_cols, encoder=encoder)
print(f"\nExported: models/ranker.txt + artifacts/feature_names.json")
for p in sorted(Path("models").glob("ranker*")):
    print(f"  {p} ({p.stat().st_size / 1024:.1f} KB)")
for p in sorted(Path("artifacts").glob("*.json")):
    print(f"  {p}")

# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 50}")
print("✅ PIPELINE COMPLETE — all steps executed successfully")
print(f"{'=' * 50}")
