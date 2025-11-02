### Betpredictor

Betpredictor predicts MLB game win probabilities and compares them to market prices to surface positive-EV bets. The system treats the median opening price across sportsbooks as a prior (bias) and trains a model to learn residual signal around that baseline. A robust ETL pipeline backs the model with clean, reproducible data.

Highlights
- End-to-end pipeline: Scrapers → SQLite DB → Feature engineering → Preprocessing → Modeling → Calibration → Backtesting.
- Market-aware modeling: Uses opening-line implied probabilities as a base margin for XGBoost, emphasizing disagreement with the market.
- Strong engineering: Cached features, reproducible splits, logging, schema auto-migrations, and modular components.


What’s in this repo
- src/scrapers: Scrapy + Playwright spiders for odds, rosters/lineups, player stats, fielding FRV, and park factors, with pipelines saving to SQLite.
- src/data/database.py: Thread‑safe SQLite manager (WAL, connection pooling) with schema utilities.
- src/scrapers/schema.sql: Canonical database schema; auto-migrated as needed.
- src/data/features: Feature generators (player, team, game context, odds) and the orchestration pipeline.
- src/data/feature_preprocessing.py: Train/val/test splits, scaling, caching to parquet.
- src/data/models: XGBoost baseline, probability calibration (Platt, isotonic, temperature), plots and saved artifacts.
- src/data/backtesting/expected_return.py: Simple EV policy backtest on test split.
- src/tools: Utilities including automated schema updates and schedule fetch via MLB StatsAPI.
- src/config.py: Project paths, season ranges (DATES), team mappings, and constants.


Prerequisites
- Python 3.10+ recommended.
- SQLite (bundled with Python) and adequate disk space (DB can be large).
- Playwright browser binaries for the scrapers that render pages.

Install
1) Create a virtual environment and activate it
- macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
- Windows (PowerShell): `py -m venv .venv; .\.venv\Scripts\Activate.ps1`

2) Install the package and core dependencies
- From repo root: `pip install -e .`

3) Install model/plotting dependencies (not listed in pyproject)
- `pip install xgboost scikit-learn optuna matplotlib scipy joblib tqdm`

4) Install Playwright browser (for scrapy-playwright)
- `python -m playwright install chromium`


Database and Data Ingestion
The project uses a single SQLite database at `src/data/mlb_stats.sqlite` defined in `src/config.py`.

- Initialize / refresh schedule (MLB StatsAPI)
  - `python -m src.tools.fetch_schedule`
  - This also ensures the schema is up to date and clears/rebuilds `schedule` and `rosters` tables.

- Run scrapers (Scrapy + Playwright)
  - Run from `src/` so that `scrapy.cfg` is picked up:
    - `cd src`
    - Odds: `scrapy crawl odds`
    - Player game logs (batting, pitching): `scrapy crawl stats`
    - Lineups + starters: `scrapy crawl lineups`
    - Fielding FRV (Baseball Savant): `scrapy crawl fielding`
    - Park factors: `scrapy crawl park_factor`
  - Notes:
    - Pipelines write directly into SQLite and will drop/recreate some tables (e.g., odds) for consistency.
    - Playwright must be installed and able to launch headless Chromium.
    - The date ranges per season are controlled in `src/config.py:DATES`.


Feature Engineering and Preprocessing
The feature pipeline merges schedule, odds, lineups, batting/pitching rolling features (EWM and season-to-date with shrinkage to league priors), fielding FRV, and game context (weather, park factors, day/night).

Primary entrypoint: `src/data/feature_preprocessing.py`
- Default run (builds features, splits, scales, and caches):
  - `python -m src.data.feature_preprocessing --log --clear-log`
  - Output cache: `src/data/features/cache/` with `X_*`, `y_*`, `odds_data` parquet files and a saved scaler.
- Options:
  - `--force-recreate`: recompute rolling features from raw tables even if cached.
  - `--force-recreate-preprocessing`: re-do splits/scale even if cached.
  - To restrict seasons, instantiate `PreProcessing([YYYY, ...]).preprocess_feats(...)` in a short script or a notebook; the CLI default uses `[2021, 2022, 2023, 2024, 2025]`.


Modeling
The baseline model is XGBoost trained on engineered features with the market’s implied probability as a base margin. This focuses learning on residual signal vs the opening line.

Entrypoint: `src/data/models/xgboost_model.py`
- Train and evaluate:
  - `python -m src.data.models.xgboost_model --log`
  - Add `--retune` to run Optuna hyperparameter search before training.
  - Add `--force-recreate` / `--force-recreate-preprocessing` to rebuild features if needed.
- Artifacts and reports:
  - Plots: `src/data/models/plots/` (ROC, calibration, residual diagnostics).
  - Hyperparameters: `src/data/models/saved_hyperparameters/`.
  - Calibrators: `src/data/models/calibrators/` (Platt, isotonic, temperature supported).
  - Models: `src/data/models/saved_models/`.

Backtesting (Expected Return)
Entrypoint: `src/data/backtesting/expected_return.py`
- Computes simple EV metrics from model probabilities vs opening prices in the test split.
- Run: `python -m src.data.backtesting.expected_return --log`
- Outputs summary statistics and can plot EV distributions (see code for toggles).


Testing
- Quick runner: `python run_tests.py --all`
- Selective examples:
  - Loaders only: `python run_tests.py --loaders`
  - Features only: `python run_tests.py --features`
  - Individual: `python run_tests.py --pipeline` (feature pipeline tests)


Configuration
- `src/config.py`
  - `DATABASE_PATH`, `SCHEMA_PATH`, cache locations.
  - `DATES`: per-season start/end; adjust here to narrow scrape windows.
  - Team and ID mappings for MLB StatsAPI and odds sources.
- `.env`: API endpoints for scrapers (Fangraphs, Baseball Savant, SBR odds).


Roadmap / Future Work
- Additional models: MLP, LSTM/temporal models, and ensembling over XGBoost.
- Richer features: travel/rest, umpire, lineup strength projections.
- Live workflow: daily incremental updates and automated report generation.
- Market coverage: totals, runline, and cross-book best-price selection.
- Advanced policies: Kelly sizing variants, portfolio constraints, and risk controls in backtests.
