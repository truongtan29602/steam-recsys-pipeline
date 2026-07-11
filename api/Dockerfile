FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

# System deps for torch + faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code — api/app/* goes directly to /app/
COPY api/app/ ./
COPY src ./src
COPY scripts ./scripts

# Copy optional runtime artifact directories as directories, not individual
# model files. Some model artifacts (for example
# outputs/task3/two_tower_enriched.pkl) are produced by notebook/pipeline runs
# and may not exist on a fresh checkout. A hard COPY of one missing artifact
# makes `docker compose build api` fail before the API can start. The API
# already handles a missing two-tower model at runtime and the compose file also
# bind-mounts these directories for local artifacts.
COPY outputs ./outputs
COPY models ./models
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
