# Steam Dataset Setup

Chosen dataset: Steam.

## Raw files to stage manually

- steam_reviews.json.gz from https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
- steam_games.json.gz from https://cseweb.ucsd.edu/~wckang/steam_games.json.gz

Place them in data/raw/steam/. Do not commit raw data.

## Future preparation decisions to document

- Positive interaction definition
- Timestamp field and time-based split boundaries
- Duplicate and missing metadata handling
- Most-recent-5M subsampling if needed
- Sparsity, interaction distribution, long tail, and temporal patterns
