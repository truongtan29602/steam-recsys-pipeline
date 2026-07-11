# ANALYSIS.md — Steam Real-Time Recommendation System

This document summarizes the final system behavior and the main insights from Tasks 1–6. The project uses the Steam implicit-feedback dataset with a mandatory time-based train/validation/test split, from-scratch retrieval models, FAISS candidate retrieval, LightGBM LambdaRank re-ranking, and a Dockerized FastAPI + Streamlit demo.

## 1. Ablation: Retrieval-only vs. Retrieval + Ranking

The most important product ablation is whether the system should serve Two-Tower candidates directly or re-rank those candidates with engineered features and LightGBM LambdaRank.

| System variant | Candidate source | Re-ranking | Recall@20 | NDCG@10 | Catalog coverage | Notes |
|---|---|---|---:|---:|---:|---|
| Retrieval only | Enriched Two-Tower + FAISS | None | 0.0803 | 0.0190 | Not measured in this run | Fast candidate generation, but weak top-10 ordering. |
| Retrieval + ranking | Enriched Two-Tower + FAISS | LightGBM LambdaRank | Candidate recall unchanged | 0.3011 | 0.0330 | Large top-10 relevance gain from ranking features. |

**Interpretation.** The Two-Tower model is useful as a fast retrieval stage: it narrows the 11.8K-item catalog to a small candidate set. However, raw embedding scores alone are not strong enough to order the first page. Adding LightGBM improves NDCG@10 from **0.0190** to **0.3011**, which is the main quality lift for the demo.

## 2. Cold Start / User Activity Analysis

User activity strongly affects recommendation quality in implicit-feedback systems. The current API handles the practical serving cases below.

| User segment | Definition | Serving behavior | Expected quality | Product treatment |
|---|---|---|---|---|
| Unknown / cold user | User ID not found by the Two-Tower model | Fall back to popularity | Stable but non-personalized | Homepage shows “Popular right now”. |
| Light-history user | Few held-out interactions | Two-Tower embedding may be noisy; fallback remains available | Moderate; benefits from popularity and item-popularity features | Use trending rows plus “Because you liked X”. |
| Heavy-history user | Many interactions in train/test periods | Personalized Two-Tower embedding + ranking features | Best personalization potential | Personalized homepage and history-driven rows. |

**Insight.** The fallback strategy is important because the Steam dataset has a long-tail user distribution. Many users have only a small number of interactions, so pure personalization can overfit or become unstable. The demo combines anonymous popularity, personalized recommendations, item-to-item similarity, and “Because you liked X” so the product remains useful even when the user vector is weak.

**Recommended final measurement.** Before the defense, compute Recall@20 and NDCG@10 by user activity buckets such as `1–2`, `3–5`, `6–10`, and `10+` positive interactions. The expected pattern is that popularity is competitive for sparse users, while Two-Tower + LightGBM is strongest for active users.

## 3. Feature Importance and Ranking Features

The ranking stage uses engineered user-level, item-level, cross, and temporal features. The current checked-in model summary does not include an exported feature-importance artifact, so this section documents the feature groups used by the ranker and their expected importance.

| Rank | Feature / feature group | Type | Why it matters |
|---:|---|---|---|
| 1 | Item popularity / interaction count | Item-level | Strong global prior in sparse implicit data. |
| 2 | Item average hours | Item-level | Separates briefly sampled games from deeply played games. |
| 3 | User activity count | User-level | Indicates confidence in the user profile. |
| 4 | User average playtime | User-level | Captures whether the user samples broadly or deeply engages. |
| 5 | User-item category affinity | Cross-feature | Measures alignment between a candidate category and user history. |
| 6 | Candidate retrieval score / embedding similarity | Cross-feature | Carries semantic relevance from Two-Tower retrieval into ranking. |
| 7 | Recency / days since last interaction | Temporal | Recent behavior better represents current intent. |
| 8 | Category one-hot features | Item metadata | Allows category-specific ranking priors. |
| 9 | Candidate playtime-derived features | Cross/item signal | Helps calibrate implicit preference strength. |
| 10 | Long-tail / item frequency transforms | Item-level | Helps control the popularity-diversity trade-off. |

**Interpretation.** The ranking model’s large NDCG gain suggests collaborative retrieval alone is insufficient for the Steam implicit signal. Feature-based ranking can mix popularity, category affinity, user behavior, and temporal context. A final polish step would be to export `ranker.feature_importance()` to a CSV or plot for the presentation.

## 4. Latency Breakdown

The API returns `latency_ms` per endpoint and the frontend displays it for every recommendation section.

| Component | Observed / available value | Online role | Notes |
|---|---:|---|---|
| Data fetch | Endpoint-specific | Fetch user history and item metadata | Current API reads parquet data; Docker stack includes PostgreSQL service/schema for production separation. |
| User tower forward pass | Included in endpoint latency | Compute user embedding | Expected to be a few ms on CPU for one user. |
| FAISS retrieval | Offline retrieval summary: ~0.3 s | Retrieve top-100 candidates | `IndexFlatIP` is reliable for the current catalog size. |
| Feature construction | Offline summary: ~10.2 s for evaluation batch | Build candidate feature rows | Best target for online caching/precomputation. |
| LightGBM ranking | Offline training: ~5.8 s; prediction included in endpoint latency | Score and sort candidates | Prediction over 100 candidates should be fast. |
| JSON + Streamlit rendering | Endpoint/frontend specific | Serialize and display cards | Rendering many cards can dominate perceived UI time. |

**Interpretation.** The architecture satisfies the assignment separation: FastAPI serves models, Streamlit calls REST endpoints, and Docker Compose runs the services. For production performance, cache train/test dataframes and precomputed user/item aggregates at startup instead of rebuilding them inside each request.

## 5. Limitations and Future Work

**Main limitation.** The saved evaluation summaries show uneven model quality: popularity is a strong Recall@20 baseline, MF-BPR performs poorly in the saved Task 3 run, and Two-Tower + ranking gives the strongest top-10 ordering but still has limited catalog coverage after ranking.

**How to address it with more time:**

1. Improve negative sampling for MF-BPR and Two-Tower, including popularity-weighted negatives and hard negatives from FAISS.
2. Export and validate LightGBM feature importance directly from the trained booster.
3. Cache API dataframes and precomputed user/item aggregates at startup to reduce latency.
4. Add diversity-aware re-ranking such as MMR to improve coverage without sacrificing too much NDCG.
5. Measure cold-start buckets quantitatively and tune separate sparse-user vs. active-user strategies.

## 6. Final Comparison Table

Metrics below come from the saved evaluation summaries and are shown on the test split unless otherwise noted.

| Method | Stage | Recall@20 | Recall@50 | NDCG@10 | Catalog coverage | Users evaluated | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Random | Baseline | 0.0012 | — | 0.0003 | 1.0000 | 592,796 | High coverage, almost no relevance. |
| Popularity | Baseline | 0.1699 | — | 0.0546 | 0.0045 | 592,796 | Strong recall baseline but recommends only head items. |
| MF-BPR | Retrieval | 0.000016 | — | 0.000007 | 0.1709 | 592,796 | From-scratch implementation present; saved run needs tuning. |
| Enriched Two-Tower | Retrieval | 0.0803 | Not in saved summary | 0.0190 | — | Evaluation sample | Good serving retriever; weaker direct ordering. |
| Two-Tower + LightGBM | Retrieval + ranking | Candidate recall inherited | — | 0.3011 | 0.0330 | Ranking evaluation sample | Best first-page relevance. |

## 7. Constraint and Policy Compliance

| Requirement / policy | Status | Evidence |
|---|---|---|
| Time-based split mandatory | Compliant | Task 1 notebook and processed train/validation/test artifacts are time-split. |
| Random and popularity baselines | Compliant | `scripts/random_rec.py`, `scripts/popular_rec.py`, and saved baseline metrics. |
| MF-BPR from scratch | Compliant | `scripts/mf_bpr.py` and Task 3 notebook. |
| Two-Tower from scratch in PyTorch | Compliant | `scripts/two_tower.py` and enriched pipeline scripts. |
| FAISS item indexing | Compliant | API builds `faiss.IndexFlatIP` from item embeddings at startup. |
| LightGBM LambdaRank | Compliant | Ranking scripts and API ranker loading path are present. |
| Demo webapp with FastAPI + Streamlit + Docker Compose | Compliant | `api/`, `frontend/`, and `docker-compose.yml`. |
| Required webapp sections | Compliant | Popularity homepage, personalized homepage, item similarity, and “Because you liked X” in `frontend/app.py`. |
| No full-pipeline recommender libraries | Compliant based on repository scan | Core models are implemented in project scripts; libraries are used for FAISS/LightGBM/utilities. |

## 8. Final Takeaway

The strongest system is a two-stage recommender: **Two-Tower + FAISS for fast candidate generation, followed by LightGBM LambdaRank for first-page ordering**. Popularity remains a necessary cold-start fallback and a strong baseline, while ranking features provide the main quality lift for the product demo.