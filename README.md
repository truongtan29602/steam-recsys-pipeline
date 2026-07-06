# Steam Real-Time Recommendation System

Documentation-first scaffold for the Steam recommender system project. No implementation code has been added yet.

## Dataset

- Chosen dataset: Steam
- Source files: UCSD Steam reviews/interactions and Steam game metadata
- Feedback type: implicit interactions/playtime
- Assignment rule: if interactions exceed 5M, keep the most recent 5M and document the strategy.

## Structure

- api/: future FastAPI service
- frontend/: future Streamlit app
- data/: local raw/interim/processed Steam data, ignored by git
- notebooks/: future Colab-compatible narrated notebooks
- src/: future reusable Python package code
- models/ and artifacts/: future local checkpoints and serving artifacts, ignored by git
- docs/: setup and planning documentation
- reports/: future figures, tables, screenshots, GIFs, and slides assets
- sql/: future PostgreSQL schema and seed assets
- docker/: future Docker support files
- tests/: future tests

## Current status

- Clean project directories created
- Steam data staging layout prepared
- Dataset manifest and documentation placeholders prepared
- No model, API, frontend, notebook code, Docker Compose, or pipeline implementation yet

See project.md for full assignment requirements.
