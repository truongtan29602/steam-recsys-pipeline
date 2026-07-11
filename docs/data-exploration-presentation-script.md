# Data Exploration Presentation Script

Use this as your spoken script for the **Task 1 / Data Exploration** part of the project defense. It is written to be clear, natural, and easy to rehearse in about **2.5–3.5 minutes**.

---

## 30-second version

For my part, I worked on the Steam data exploration and preparation. The goal was to transform raw Steam reviews and game metadata into clean implicit-feedback data for recommendation models. We kept valid user–game interactions, removed unusable rows, deduplicated repeated reviews by keeping the latest one, and defined a positive interaction as at least one hour of playtime. Then we created a chronological split with **3.6M train**, **0.4M validation**, and **1.0M test** interactions, so evaluation reflects future user behavior instead of leaking information through a random split. The main EDA findings were that playtime is extremely skewed, users and items follow a strong long-tail distribution, and item metadata is important for both the demo and ranking features. These findings explain why the final system uses popularity fallback, two-tower retrieval, and LightGBM reranking.

---

## Full slide-by-slide script

## Slide 1 — Data exploration goal

**What to say:**

“My part of the project was data exploration and preparation. The raw dataset comes from Steam reviews and Steam game metadata. The goal was to convert that raw data into something usable by our recommender system: clean user–item interactions, a positive label, chronological train/validation/test splits, and a catalog table with game metadata.

The important point is that this step is not just preprocessing. It defines the recommendation problem. By looking at playtime, user activity, item popularity, and timestamps, we can understand what kind of model architecture makes sense later.”

**Key points to emphasize:**

- Raw Steam reviews become implicit-feedback interactions.
- Data exploration guides modeling choices.
- Output is used by retrieval, ranking, API serving, and frontend display.

---

## Slide 2 — Cleaning and labeling

**What to say:**

“The raw reviews are not standard JSON; they are gzip-compressed line-delimited Python literals, so we parse them carefully. During cleaning, we keep only rows with a valid user id, item id, timestamp, and non-negative playtime.

If a user reviewed the same game more than once, we keep the latest review because it is the most recent signal. We also normalize ids as strings so the same item or user is represented consistently across the pipeline.

For the label, we use a simple implicit-feedback rule: if the user played at least one hour, `is_positive` is true. This is easy to explain and works well for Steam because playtime is a direct signal of engagement.”

**Key points to emphasize:**

- Valid rows only: user, item, date, hours.
- Latest user–item review kept.
- Positive label: `hours >= 1.0`.
- Extra features preserved: review text length, early access, products owned, funny votes, free-copy flag.

---

## Slide 3 — Chronological split

**What to say:**

“For evaluation, we use a chronological split instead of a random split. This is important because a recommender is normally trained on past behavior and evaluated on future behavior. If we split randomly, the model can indirectly learn from future interactions and the metrics become too optimistic.

Our final processed data has **5 million interactions**. We split it into **3.6 million training interactions**, **400 thousand validation interactions**, and **1 million test interactions**. The validation and test sets come later in time, so the evaluation better matches a real deployment scenario.”

**Key points to emphasize:**

- 5.0M total interactions.
- 3.6M train / 0.4M validation / 1.0M test.
- Time-based split prevents future leakage.
- This is required for honest recommender evaluation.

**Graph to show:**

- Chronological split sizes bar chart.

---

## Slide 4 — Playtime distribution

**What to say:**

“One of the most important EDA findings is that playtime is extremely skewed. In our sample, the median playtime is around **14 hours**, but the 90th percentile is around **221 hours**, and the 99th percentile is around **1,664 hours**.

This means playtime is useful, but raw hours should be handled carefully. A small number of users can have very large values, so later ranking features should use robust transformations like logs, averages, or ratios instead of relying only on raw hours.

The positive rate is also high, around **89%** in the EDA sample, which makes sense because users usually review games after actually playing them.”

**Key points to emphasize:**

- Positive rate around 89%.
- Median ≈ 14 hours.
- p90 ≈ 221 hours.
- p99 ≈ 1,664 hours.
- Skew motivates robust feature engineering.

**Graphs to show:**

- Positive rate by split.
- Log playtime distribution.

---

## Slide 5 — Long-tail behavior

**What to say:**

“The dataset also has the classic recommender-system long-tail problem. In the training sample, the median user has only one interaction, and even the 90th percentile user has only two interactions. That means many users have very sparse histories.

Items are long-tailed too. Many games have only a few interactions, while a small number of popular games receive much more activity. This creates two challenges: cold-start or weak-history users, and popularity bias toward already popular games.

This finding directly motivates our product design. For anonymous users or weak profiles, popularity is a reliable fallback. For known users, we need embedding-based retrieval and ranking features to personalize beyond just popular games.”

**Key points to emphasize:**

- Most users have very few observed interactions.
- Items follow a popularity long tail.
- Popularity is a strong baseline/fallback.
- Personalization needs embeddings and ranking.

**Graph to show:**

- Long-tail activity chart.

---

## Slide 6 — Catalog metadata

**What to say:**

“The processed item catalog contains **12,146 Steam games**. For each game we keep metadata like title, category, genres, tags, price, release date, early access status, developer, and image URL.

This metadata matters for two reasons. First, it makes the frontend demo readable because we can show game titles, categories, and images instead of just item ids. Second, it improves modeling because the two-tower and ranking stages can use item features like category, price, release year, and tags.

So the catalog is not just for display. It connects data preparation to model quality and product quality.”

**Key points to emphasize:**

- 12,146 catalog games.
- Metadata supports UI and model features.
- Enables explainability and better ranking.

**Graph to show:**

- Top categories chart.

---

## Slide 7 — Modeling implications

**What to say:**

“To conclude, the data exploration shaped the final recommender architecture.

Because user histories are sparse, we need popularity fallback for guests and cold users. Because item popularity is long-tailed, we need to evaluate coverage and avoid only recommending the same popular games. Because playtime is skewed, we need robust feature engineering. Because the split is chronological, our validation and test metrics are closer to future production behavior. And because we have rich metadata, we can build a better demo and stronger ranking features.

So Task 1 produced the foundation for the entire project: clean interaction data, honest evaluation splits, and a feature-rich catalog used by retrieval, ranking, and serving.”

**Key points to emphasize:**

- Sparse histories → popularity fallback.
- Long-tail items → coverage and personalization.
- Skewed hours → robust features.
- Temporal split → honest evaluation.
- Metadata → better ranking and demo.

---

## Natural transition to the next speaker

“After this data preparation step, the next part of the project can train baseline recommenders and personalized retrieval models on clean chronological splits. That is where we compare popularity, MF-BPR, and the two-tower retrieval model before adding LightGBM ranking.”

---

## If you get asked questions

### Why use `hours >= 1` as positive?

“Because Steam playtime is a direct engagement signal. One hour is simple, explainable, and filters out very short accidental interactions while keeping enough positives for training.”

### Why not random split?

“Random split leaks future behavior into training. A chronological split better simulates production, where we train on past interactions and recommend for future ones.”

### Why is popularity still important?

“The EDA shows many users have very sparse histories. For anonymous or weak-history users, popularity is often the safest fallback and a strong baseline.”

### Why keep metadata?

“Metadata supports both the user-facing demo and model features. It lets us display readable cards and gives the ranking model signals like category, price, and release information.”
