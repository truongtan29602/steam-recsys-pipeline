# Data Directory

Local data staging only. Dataset files are ignored by git.

## Layout

- `raw/steam/`: original downloaded Steam files
- `interim/steam/`: cleaned intermediate files generated later
- `processed/steam/`: train/validation/test and serving-ready outputs generated later
- `external/`: optional third-party reference files

## Expected raw files

- `data/raw/steam/steam_reviews.json.gz`
- `data/raw/steam/steam_games.json.gz`
