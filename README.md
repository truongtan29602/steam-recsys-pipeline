# Steam Real-Time Recommendation System

Portfolio-ready recommender system for the EPITA AIS3 project: raw Steam implicit-feedback data → retrieval and ranking models → FastAPI model serving → Streamlit webapp demo.

## What is implemented

| Project task | Status | Main files |
|---|---|---|
| 1. Data preparation & exploration | Done | `notebooks/01_task1_data_preparation_exploration.ipynb`, `src/steam_recsys/data_prep.py` |
| 2. Baselines | Done | `scripts/random_rec.py`, `scripts/popular_rec.py`, `scripts/evaluate_baselines.py` |
| 3. Retrieval | Done | `scripts/mf_bpr.py`, `scripts/two_tower.py`, `notebooks/03_task3_matrix_factorization.ipynb` |
| 4. Demo webapp / ranking pipeline | Done + polished | `api/app/main.py`, `frontend/app.py`, `docker-compose.yml`, `src/steam_recsys/ranking/` |
| 5. Analysis | Done | `ANALYSIS.md` |
| 6. Notebook quality/documentation | Done by team sync | `notebooks/` |

## Architecture

```text
Offline training / evaluation
Steam data -> time split -> baselines -> MF-BPR + Two-Tower -> FAISS candidates -> LightGBM ranking

Online demo
Streamlit frontend -> FastAPI REST API -> model artifacts + FAISS index + item metadata -> recommendations
Docker Compose services: postgres, api, frontend
```

Required product sections are implemented in `frontend/app.py`:

1. **Guest homepage**: popularity recommendations.
2. **Logged-in homepage**: Two-Tower retrieval + LightGBM ranking.
3. **Item page similar items**: cosine similarity over learned item embeddings.
4. **Because you liked X**: nearest neighbors for an item from the user history.

## Key results

| Method | Recall@20 | NDCG@10 | Catalog coverage | Notes |
|---|---:|---:|---:|---|
| Random | 0.0012 | 0.0003 | 1.0000 | Sanity-check baseline. |
| Popularity | 0.1699 | 0.0546 | 0.0045 | Strong recall but low diversity. |
| MF-BPR | 0.000016 | 0.000007 | 0.1709 | From-scratch model; saved run needs tuning. |
| Enriched Two-Tower | 0.0803 | 0.0190 | — | Fast retrieval stage. |
| Two-Tower + LightGBM | Candidate recall inherited | 0.3011 | 0.0330 | Best top-10 ranking quality. |

See `ANALYSIS.md` for the ablation table, cold-start discussion, feature importance discussion, latency breakdown, limitations, and final comparison.

## Run the demo

From the repository root:

```bash
docker compose up --build
```

Then open:

- Streamlit frontend: <http://localhost:8501>
- FastAPI Swagger docs: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>

> Note: the API expects processed data/model artifacts under the paths mounted in `docker-compose.yml` (`data/`, `models/`, `artifacts/`, `outputs/`). If large model artifacts are stored outside Git, restore them before running the full live demo.

## Local development checks

```bash
python3 -m py_compile frontend/app.py api/app/main.py scripts/mf_bpr.py scripts/two_tower.py
```

## Repository structure

```text
api/                         FastAPI recommendation service
frontend/                    Streamlit UI
notebooks/                   Colab-readable narrated notebooks
src/steam_recsys/            Reusable package code
scripts/                     Training/evaluation/serving utilities
sql/init/                    PostgreSQL initialization schema
data/                        Local/raw/processed data area
outputs/                     Saved evaluation outputs and task artifacts
models/, artifacts/          Serving artifacts and documentation placeholders
docs/                        Architecture and task reports
ANALYSIS.md                  Required analysis deliverable
docker-compose.yml           postgres + api + frontend stack
```

## Constraint compliance summary

- Time-based split is used for train/validation/test.
- Random and popularity baselines are implemented.
- MF-BPR training loop is implemented from scratch.
- Two-Tower model is implemented in PyTorch project code.
- FAISS is used for item embedding retrieval.
- LightGBM LambdaRank is used for ranking.
- No full-pipeline recommender library is used for the core models.
- The demo is separated into PostgreSQL, FastAPI, and Streamlit services via Docker Compose.

## Team task split

- Task 1–3: completed by teammates and synced into this branch.
- Task 4: ranking/webapp sync plus frontend polish on `synchronization`.
- Task 5: `ANALYSIS.md` completed/refined on `synchronization`.
- Task 6: notebook deliverable synced from team work.

## Demo media

Place the final GIF or screen recording under `reports/media/` and embed it here before submission if required by the instructor.