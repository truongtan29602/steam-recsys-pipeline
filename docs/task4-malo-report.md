# Task 4 — Ranking: Malo's Report

> **Malo Fargeas** — EPITA AIS3 — Steam Recommendation System Project
>
> Team: Kim Tan (lead, Task 1), Hamza (Task 3), Linh Long (baselines), Malo (Task 4)
>
> Date: July 10, 2026

---

## 1. Task Objective

Build a **LightGBM LambdaRank reranker** that takes retrieval candidates and re-orders them to improve recommendation quality.

**What the rubric requires:**
- 10+ distinct features (user, item, cross, temporal)
- LightGBM LambdaRank trained on retrieval candidates
- NDCG@10, Catalog Coverage
- Retrieval-only vs retrieval+ranking comparison
- Feature importance, latency breakdown

---

## 2. Ranking Pipeline Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Retrieval   │────▶│ Feature Engine  │────▶│  LightGBM    │
│  (top-100)   │     │  14 → 71 feats  │     │  LambdaRank  │
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                    │
                                              ┌─────▼──────┐
                                              │  Top-10    │
                                              │  re-ranked │
                                              └────────────┘
```

### Features (14 logical → 71 after one-hot encoding)

| Group | # | Features |
|--------|---|----------|
| **User** | 6 | activity_count, avg_hours, distinct_items, positive_rate, total_hours, avg_text_length |
| **Item** | 5 | popularity, avg_hours, positive_rate, avg_text_length, is_early_access |
| **Cross** | 2 | user_category_affinity, user_item_hours_vs_avg |
| **Temporal** | 3 | days_since_last, hour_sin, hour_cos |
| **One-hot** | 55 | 60 Steam categories (Action, RPG, Indie, etc.) |

---

## 3. Experiment History

### Experiment 1: Synthetic smoke test
**When:** July 10, morning
**Goal:** Validate the pipeline works before real data is available.
**Setup:** 500 users, 1000 items, 200K randomly generated interactions.
**Result:** Pipeline OK — 22 features, LightGBM 0.1s training, NDCG 1.0 (trivial — random data is too easy).

### Experiment 2: Real data with random candidates
**When:** July 10, afternoon
**Goal:** First run on real Steam data (3.6M interactions).
**Setup:** Simulated candidates (1 positive + 99 random negatives per user).
**Result:** NDCG@10 = 0.7262, Catalog Coverage = 25.6%.
**Issue:** Random negatives are too easy — NDCG overestimated.

### Experiment 3: Popularity → LightGBM (clean evaluation)
**When:** July 10, evening
**Goal:** Clean pipeline using Hamza's PopularityRecommender, no circularity.
**Setup:** 500 val/test users, Popularity top-100, labels = ground truth.
**Result:**
| | Retrieval | +Ranking |
|---|-----------|----------|
| NDCG@10 | 5.3% | 95.2% |
**Issue:** NDCG 95% = data leakage. Negatives had `hours=0` → the `user_item_hours_vs_avg` feature trivially separated positives from negatives.

### Experiment 4: Enriched Two-Tower (our features)
**When:** July 10, night
**Goal:** Fix Hamza's broken two-tower (0% recall) by adding content features.
**Changes:**
- Two-tower: 43 item features (15 genres, 20 tags, price, release year, developer, early access) instead of 4
- Architecture: Linear(43,256)→ReLU→Dropout→Linear(256,128)→ReLU→Linear(128,64)
- Vectorized feature building (groupby agg + reindex, 10× faster than the original Python loop)
- 15 epochs, batch 4096, learning rate 5e-4
**Retrieval result:** Recall@20 = 8.0%, NDCG@10 = 1.9% (3000× better than Hamza's two-tower)

### Experiment 5: Enriched Two-Tower → LightGBM (leakage fixed)
**When:** July 10, night
**Goal:** Full pipeline with enriched two-tower, no data leakage.
**Fix:** Negatives → `hours = item_avg_hours` (item's average playtime from training) instead of 0.
**Final result (honest):**

| | Retrieval | +Ranking |
|---|-----------|----------|
| NDCG@10 | 1.90% | **30.11%** |
| Recall@20 | 8.03% | — |
| Catalog Coverage | — | 3.30% |

**Improvement: +28.2 NDCG points.** Ranking improves quality by 15×.

---

## 4. Final Results — Comparison Table

```
═════════════════════════════════════════════════════
                 Recall@20   NDCG@10    Source
─────────────────────────────────────────────────────
Random            0.12%       0.03%    Hamza
Popularity       16.99%       5.46%    Hamza
MF-BPR            0.002%      0.0008%  Hamza
Two-Tower basic   0.002%      0.0003%  Hamza
Two-Tower enriched 8.03%      1.90%    OURS
TT enriched + LGBM    —      30.11%    OURS (ranking)
═════════════════════════════════════════════════════
```

---

## 5. Bugs Found and Fixed

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| 1 | `evaluate_ranker` counted numpy indices instead of item IDs for Catalog Coverage | 🔴 Critical | Added `item_ids` parameter, tracking `str(item_ids[offset+rel_idx])` |
| 2 | `fillna(0)` before LightGBM destroyed native NaN handling | 🟠 Major | Removed, NaN passes through to LightGBM |
| 3 | `df.get("hours", ...)` fallback never triggered for NaN values | 🟠 Major | Replaced with `df["hours"].fillna(df["user_avg_hours"])` |
| 4 | `item_popularity_rank` redundant with `item_popularity` (corr 0.98) | 🟡 Moderate | Removed |
| 5 | `pd.get_dummies` not portable to serving | 🟡 Moderate | Replaced with `OneHotEncoder(handle_unknown="ignore")`, saved as pickle |
| 6 | **Hours leakage**: negatives had `hours=0` → `user_item_hours_vs_avg` was trivial | 🔴 Critical | Negatives now use `hours = item_avg_hours` |
| 7 | Two-tower feature building: Python `get_group()` loop over 1.5M users = 30 min | 🟠 Major | Vectorized: `groupby.agg().reindex()` = 3 min |

---

## 6. Code Produced

| File | Purpose |
|------|---------|
| `src/steam_recsys/ranking/features.py` | 14 features → 71 after one-hot encoding |
| `src/steam_recsys/ranking/train.py` | LightGBM LambdaRank + evaluation |
| `scripts/pipeline_two_tower.py` | Two-Tower → FAISS → LightGBM, full pipeline |
| `scripts/pipeline_clean.py` | Popularity → LightGBM, clean evaluation |
| `scripts/run_ranking_real.py` | Executable notebook on real data |
| `scripts/two_tower.py` | Enriched two-tower (43 item features) |
| `docs/code-review-task4.md` | Exhaustive code review (14 sections, 35 KB) |

---

## 7. Key Design Decisions

- **OneHotEncoder over pd.get_dummies**: portable to serving, `handle_unknown='ignore'` for unseen categories
- **Feature engineering on training data only**: zero leakage from val/test splits
- **LightGBM without fillna**: LightGBM handles NaN natively (learns optimal split direction)
- **Cyclic encoding for hour**: sin/cos avoids the 23h→0h discontinuity
- **Catalog coverage on real item IDs**: not numpy matrix indices
- **Enriched two-tower with content features**: 15 genres + 20 tags + price + release year + developer → enables generalization to new games

---

## 8. Lessons Learned

1. **Time-based split makes retrieval very hard.** Games played during val/test periods are often new — collaborative models (MF-BPR, basic two-tower) fail at 0%.
2. **Content features unlock the two-tower.** Going from 4 to 43 features (genres, tags, price) boosted recall from 0% to 8%.
3. **Ranking compensates for weak retrieval.** Even with 1.9% retrieval NDCG, LightGBM reaches 30% by re-ordering candidates.
4. **Watch out for data leakage.** `hours=0` for negatives creates a trivial feature that artificially inflates scores. Always use estimated values (averages) for unknown features at serving time.
5. **Pandas vectorization is critical.** A `get_group()` loop over 1.5M users = 30 minutes. `groupby.agg().reindex()` = 3 minutes.
