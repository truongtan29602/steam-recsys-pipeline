# Complete Pipeline — Steam Recommendation System

## Mermaid: Full Architecture (Offline Training + Online Serving)

```mermaid
flowchart TB
    subgraph DATA["📦 Data Preparation (Task 1 — Kim Tan)"]
        RAW[("steam_reviews.json.gz<br/>6.8M interactions")]
        GAMES[("steam_games.json.gz<br/>32K games")]
        CLEAN[("Clean + subsample<br/>→ 5M most recent")]
        SPLIT[("Time-based split<br/>72% train / 8% val / 20% test")]
        RAW --> CLEAN --> SPLIT
        GAMES --> CLEAN
    end

    subgraph RETRIEVAL["🔍 Retrieval (Task 3 — Hamza + Malo)"]
        direction TB
        TRAIN_DATA[("train.parquet<br/>3.6M interactions<br/>1.49M users, 11.8K items")]
        CATALOG[("items.parquet<br/>15.4K games<br/>60 categories")]
        
        subgraph TWOTOWER["Two-Tower Model"]
            UF[("User Features<br/>8 dims: count, avg_hours,<br/>diversity, price, ea_rate...")]
            IF[("Item Features<br/>43 dims: 15 genres, 20 tags,<br/>price, year, dev, ea...")]
            UT["User Tower<br/>Linear(8→256→128→64)<br/>+ ReLU + Dropout"]
            IT["Item Tower<br/>Linear(43→256→128→64)<br/>+ ReLU + Dropout"]
            UE[("User Embeddings<br/>64d, computed at runtime")]
            IE[("Item Embeddings<br/>64d, pre-computed<br/>11,808 vectors")]
            UF --> UT --> UE
            IF --> IT --> IE
        end

        FAISS[("FAISS IndexFlatIP<br/>11,808 vectors × 64d<br/>inner product search")]
        TOP100[("Top-100 candidates<br/>per user")]
        
        TRAIN_DATA --> UF
        TRAIN_DATA --> IF
        CATALOG --> IF
        IE --> FAISS
        FAISS -->|"user_emb @ item_embs"| TOP100
    end

    subgraph RANKING["🎯 Ranking (Task 4 — Malo)"]
        direction TB
        FEAT[("Feature Engineering<br/>14 features:<br/>user(6) + item(5) + cross(2) + temporal(3)<br/>→ 71 after OHE")]
        LGBM[("LightGBM LambdaRank<br/>40 trees, 0.6s train<br/>NDCG: retrieval 1.9% → 30.1%")]
        TOP10[("Top-10 re-ranked<br/>recommendations")]
        
        TOP100 --> FEAT --> LGBM --> TOP10
    end

    subgraph SERVING["🌐 Online Serving (Webapp)"]
        direction TB
        PG[("PostgreSQL<br/>user profiles, history,<br/>item metadata")]
        API["FastAPI<br/>loads models at startup<br/>Swagger UI at /docs"]
        FRONT["Streamlit<br/>sections:<br/>• Guest → Popularity<br/>• Logged in → Two-Tower+LGBM<br/>• Similar items → Cosine<br/>• Because you liked X → Cosine"]
        PG --> API --> FRONT
    end

    SPLIT -->|"train.parquet"| TRAIN_DATA
    SPLIT -->|"val.parquet"| RETRIEVAL
    SPLIT -->|"test.parquet"| RANKING 

    UE -->|"at request time"| FAISS
    TOP10 -->|"serve top-10"| API
    IE -->|"pre-load"| API
    LGBM -->|"models/ranker.txt"| API
    CATALOG -->|"seed DB"| PG
    SPLIT -->|"seed DB"| PG
```

## Mermaid: Training Flow (Offline)

```mermaid
flowchart LR
    subgraph STEP1["Step 1: Two-Tower Training"]
        T1[("train.parquet")] --> T2["Build user features (8d)"]
        T1 --> T3["Build item features (43d)"]
        T2 --> T4["Train Two-Tower<br/>30 epochs, GPU<br/>in-batch negatives<br/>softmax cross-entropy"]
        T3 --> T4
        T4 --> T5[("two_tower_enriched.pkl<br/>89 MB")]
    end

    subgraph STEP2["Step 2: FAISS Indexing"]
        T5 --> F1["Extract item_embeddings<br/>(11808, 64)"]
        F1 --> F2["Build IndexFlatIP"]
        F2 --> F3[("FAISS ready")]
    end

    subgraph STEP3["Step 3: Candidate Generation"]
        F3 --> C1["For each val/test user:<br/>compute user_emb → FAISS top-100"]
        C1 --> C2[("retrieval candidates<br/>50K rows, 500 users")]
    end

    subgraph STEP4["Step 4: LightGBM Training"]
        C2 --> L1["14 features → 71 after OHE"]
        L1 --> L2["Train LambdaRank<br/>500 rounds, early stopping"]
        L2 --> L3[("models/ranker.txt<br/>500 KB")]
    end

    T5 --> STEP2
```

## Mermaid: Online Serving Flow (Single Request)

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant S as 🖥️ Streamlit
    participant A as ⚡ FastAPI
    participant D as 🗄️ PostgreSQL
    participant T as 🧠 Two-Tower
    participant F as 🔍 FAISS
    participant L as 🎯 LightGBM

    U->>S: Open "Logged in" page
    S->>A: GET /recommendations/personalized/{user_id}
    A->>D: SELECT history WHERE user_id = ?
    D-->>A: user history + item catalog
    A->>T: user_tower.forward(user_features)
    T-->>A: user_embedding (64d)
    A->>F: index.search(user_emb, k=100)
    F-->>A: top-100 item_ids
    A->>A: Feature engineering (14 features)
    A->>L: model.predict(feature_matrix)
    L-->>A: scores, top-10 indices
    A->>D: SELECT title, image, category WHERE item_id IN (...)
    D-->>A: item metadata
    A-->>S: JSON: {section, model, items[]}
    S-->>U: 🎮 Game cards with titles + images
```

## Key Numbers

| Stage | Metric | Value |
|-------|--------|-------|
| **Data** | Train interactions | 3.6M |
| | Users / Items | 1.49M / 11.8K |
| | Sparsity | 99.98% |
| **Two-Tower** | Item features | 43 (15 genres + 20 tags + 8 stats) |
| | User features | 8 |
| | Embedding dim | 64 |
| | Training time | 411s (GPU) |
| | Recall@20 | 8.0% |
| **FAISS** | Index size | 11,808 vectors |
| | Search time | ~0.3ms |
| **LightGBM** | Features after OHE | 71 |
| | Trees | 40 |
| | Training time | 0.6s |
| | NDCG@10 improvement | 1.9% → 30.1% (+28.2 pts) |
| **Serving** | Total latency | ~10ms |
| | Model size (total) | ~90 MB |
