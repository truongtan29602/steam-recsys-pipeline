# Data Exploration Slides — Content + LLM Prompt

Use this file as source content or as a prompt for another LLM / slide generator.

## Role + Goal Prompt

You are creating a polished 5–7 slide presentation section for a recommender-system project defense. The section is **Task 1: Steam data exploration and preparation**. The audience is technical but time-constrained: instructors and classmates who need to understand why the dataset is sparse, long-tailed, temporally split, and suitable for a two-stage recommender system.

Create slides with a clean dark Steam-inspired theme: navy/black background, cyan/green accents, readable white text, and large chart callouts. Keep slides concise, but include speaker notes for a 2–3 minute oral explanation.

Use the graphs generated in `slides/data_exploration/figures/`:

1. `01_split_sizes.png` — chronological split sizes
2. `02_positive_rate_by_split.png` — positive interaction rate by split
3. `03_log_playtime_distribution.png` — log playtime distribution
4. `04_long_tail_activity.png` — user/item long-tail activity
5. `05_top_categories.png` — catalog category mix

## Key Facts to Use

- Dataset source: UCSD Steam reviews + Steam games metadata.
- Raw format: gzip-compressed line-delimited Python literals, parsed safely with `ast.literal_eval`.
- Final interaction dataset: **5,000,000 interactions**.
- Chronological split:
  - Train: **3,600,000 interactions**
  - Validation: **400,000 interactions**
  - Test: **1,000,000 interactions**
- Catalog metadata: **12,146 Steam games** with known category metadata in the processed catalog.
- Interaction columns used downstream:
  - `user_id`, `item_id`, `hours`, `event_time`, `is_positive`, `text_length`, `early_access_review`, `products_owned`, `found_funny`, `received_for_free`
- Item metadata columns:
  - `item_id`, `title`, `category`, `genres`, `tags`, `price`, `release_date`, `early_access_item`, `developer`, `image_url`
- Positive interaction definition: **hours >= 1.0**.
- Sample-based EDA summary (650k deterministic sample for fast plotting):
  - Sample positive rate: **89.16%**
  - Train sample positive rate: **88.81%**
  - Validation sample positive rate: **88.41%**
  - Test sample positive rate: **90.33%**
  - Sample median playtime: around **14 hours**
  - Sample p90 playtime: around **221 hours**
  - Sample p99 playtime: around **1,664 hours**
- Long-tail behavior in train sample:
  - Median sampled user has **1 interaction**
  - 90th percentile sampled user has **2 interactions**
  - 99th percentile sampled user has **6 interactions**
  - Median sampled item has **5 interactions**
  - 90th percentile sampled item has about **70 interactions**
  - 99th percentile sampled item has about **533 interactions**
- Time windows observed in sample:
  - Train: **2015-04-12 → 2017-06-25**
  - Validation: **2017-06-25 → 2017-08-28**
  - Test: **2017-08-28 → 2018-01-05**

> Note for slide wording: user/item uniqueness and quantiles above are sample-based except row counts and catalog size. Phrase them as “sample shows…” or “EDA sample indicates…” unless using exact split row counts.

---

## Slide 1 — Title / Why Data Exploration Matters

**Title:** Steam RecSys Data Exploration: turning reviews into recommendation signals

**Main message:** We convert raw Steam reviews into implicit-feedback interactions and validate that the data has the key RecSys challenges: sparsity, skewed playtime, long-tail users/items, and temporal drift.

**Bullets:**
- Source: Steam reviews + game metadata.
- Target: model-ready implicit feedback for retrieval, ranking, and serving.
- Main design choice: use playtime before review as an implicit preference signal.
- Output: chronological train/validation/test splits + item catalog.

**Visual suggestion:** Small pipeline diagram:
`Raw Steam reviews + games → Cleaning → Positive labels → Time split → Model-ready parquet + catalog`

**Speaker notes:**
“My part was data exploration and preparation. The key was not just loading the data, but checking whether it behaves like a real recommender-system dataset. Steam reviews contain user, game, date, and playtime signals. We turn those into implicit interactions where at least one hour of playtime is considered positive. Then we create chronological splits to evaluate the recommender as if it were deployed in the future.”

---

## Slide 2 — Dataset and Cleaning Decisions

**Title:** From messy raw files to clean interaction tables

**Main message:** The data preparation standardizes users/items, removes invalid rows, deduplicates repeated user–game reviews, and keeps features useful for ranking.

**Bullets:**
- Raw reviews are gzip line-delimited Python literals, not strict JSON.
- Kept only rows with valid `user_id`, `item_id`, `event_time`, and non-negative `hours`.
- Duplicate user–item reviews are collapsed by keeping the latest signal.
- Positive label: `is_positive = hours >= 1.0`.
- Preserved side features: text length, early-access review flag, products owned, funny votes, free-copy flag.

**Visual suggestion:** A compact “cleaning checklist” with icons.

**Speaker notes:**
“The raw format is a bit unusual, so parsing had to be robust. We remove rows that cannot be used for recommendation, normalize ids as strings, and use the latest review when a user reviewed the same game more than once. The one-hour threshold is a simple, explainable implicit signal: it separates very short interactions from games the user actually tried.”

---

## Slide 3 — Chronological Split for Honest Evaluation

**Title:** We split by time, not randomly

**Main message:** A recommender should be evaluated on future behavior, so the project uses a chronological train/validation/test split.

**Use figure:** `01_split_sizes.png`

**Bullets:**
- Total: **5.0M interactions**.
- Train: **3.6M** older interactions.
- Validation: **0.4M** later interactions for model selection.
- Test: **1.0M** newest interactions for final reporting.
- Prevents leakage from future user behavior into training.

**Speaker notes:**
“A random split would leak future preferences into training and overestimate model quality. We instead sort interactions by event time. The train set contains older behavior, validation is the next time window, and test is the newest period. This makes Task 3 and Task 4 evaluation closer to the real production problem.”

---

## Slide 4 — Implicit Feedback: Playtime Is Skewed

**Title:** Playtime is a strong but highly skewed preference signal

**Use figures:** `02_positive_rate_by_split.png` and `03_log_playtime_distribution.png`

**Bullets:**
- Most reviews have at least one hour of playtime: sample positive rate around **89%**.
- Median playtime is around **14 hours**, but p90 is around **221 hours**.
- The tail is very large: p99 is around **1,664 hours**.
- This motivates robust features like `log1p(hours)` or item/user averages, not raw hours alone.

**Speaker notes:**
“The positive rate is high because Steam reviews are usually written after actually playing a game. But the distribution is extremely skewed: some users have hundreds or thousands of hours. That means raw hours can dominate models, so later ranking features should use normalized or aggregate signals.”

---

## Slide 5 — Long-Tail Users and Items

**Title:** The dataset has classic recommender long-tail behavior

**Use figure:** `04_long_tail_activity.png`

**Bullets:**
- In the train sample, the median user appears with **1 interaction**.
- Even the 90th percentile sampled user has only **2 interactions**.
- Items are also long-tailed: many games have few observations, while a small set is highly popular.
- This creates cold-start and sparse-history challenges.

**Main implication callout:** “Popularity is a strong fallback, but personalization needs embeddings and ranking features.”

**Speaker notes:**
“This explains why our final demo includes popularity for guests and fallback cases. Many users have too little history for pure personalization. At the same time, popular items receive much more data than niche games, so retrieval and ranking must deal with popularity bias.”

---

## Slide 6 — Catalog Metadata Enables Enriched Recommendations

**Title:** Item metadata makes the system explainable and rankable

**Use figure:** `05_top_categories.png`

**Bullets:**
- Processed catalog: **12,146 Steam games**.
- Metadata includes title, category, genres, tags, price, release date, early access, developer, and image URL.
- Metadata supports:
  - richer Streamlit cards,
  - item features for the Two-Tower model,
  - category/price/release features for LightGBM ranking,
  - explainability in the demo.

**Speaker notes:**
“The catalog is important because the final product is not just ids. Metadata lets us show readable game cards in the UI and gives the ranker features beyond interaction counts. This is how data preparation connects directly to the demo quality.”

---

## Slide 7 — Takeaways for Modeling

**Title:** EDA findings shaped the recommender architecture

**Bullets:**
- **Sparse user histories** → need popularity fallback and robust cold-user behavior.
- **Long-tail item popularity** → evaluate coverage and avoid relying only on top games.
- **Skewed playtime** → use thresholds, logs, and aggregate features.
- **Temporal split** → validation/test reflect future recommendation quality.
- **Metadata-rich catalog** → enables enriched two-tower retrieval and LightGBM reranking.

**Closing line:** “Task 1 produced the clean, time-aware, feature-rich foundation used by every later model and by the live demo.”

**Speaker notes:**
“The main point is that EDA was not isolated. It directly determined the architecture: popularity fallback for sparse users, two-tower embeddings for candidate generation, LightGBM for ranking with robust features, and a product demo that can show meaningful metadata.”

---

## One-Slide Ultra-Short Version

If you only have one slide for data exploration, use this:

**Title:** Data exploration: why Steam needs a two-stage recommender

**Bullets:**
- Cleaned raw Steam reviews into **5.0M implicit-feedback interactions** and **12.1K game catalog rows**.
- Positive label is explainable: `hours >= 1`.
- Used chronological split: **3.6M train / 0.4M validation / 1.0M test** to avoid future leakage.
- EDA sample shows high positive rate (**~89%**) but extremely skewed playtime: median **~14h**, p90 **~221h**, p99 **~1,664h**.
- Long-tail behavior: most users have very few interactions, while a small set of games dominates activity.
- Implication: combine popularity fallback, embedding retrieval, metadata features, and LightGBM reranking.

**Use visual:** Combine `01_split_sizes.png` + `04_long_tail_activity.png`.
