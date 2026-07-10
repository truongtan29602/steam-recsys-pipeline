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
COPY outputs/task3/two_tower_enriched.pkl ./outputs/task3/
COPY models ./models
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
