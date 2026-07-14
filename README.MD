# Sentiment Analysis Scanner

An end-to-end pipeline that collects stock prices and financial news, scores headline sentiment, and stores everything in SQLite as a clean dataset for modeling.

```
Scrape news + prices  →  Store RAW  →  Score sentiment (versioned)  →  Query / Export / Visualize
```

Two design rules the whole project follows:

1. **Raw data is sacred.** Headlines are stored exactly as published (HTML stripped, nothing else). Preprocessing belongs to models, not to storage — so history can always be re-scored by a better model.
2. **Scores are versioned.** Sentiment lives in its own table keyed by `(article, model_name, model_version)`. Swapping VADER for a trained classifier means adding rows, never rewriting them.

## Quickstart

```bash
pip install -e ".[dev,ml]"   # ml extra = TensorFlow + scikit-learn

sent-trader scan NVDA AAPL   # scrape news + prices, score sentiment (VADER)
sent-trader train            # train the TF classifier on Financial PhraseBank
sent-trader benchmark        # TF classifier vs VADER on the held-out test set
sent-trader scan NVDA --model tf   # score with the trained classifier
sent-trader backfill         # FNSPID historical news + full price history (top-50 universe)
sent-trader score --model tf # score any unscored backlog
sent-trader forecast-train      # train the quantile forecaster
sent-trader forecast-benchmark  # coverage test vs empirical-quantile baselines
sent-trader forecast NVDA       # tomorrow's return distribution for a ticker
sent-trader list             # tickers in the database
sent-trader export           # dump prices and articles to CSV
```

The database is a single SQLite file (`sentiment.db` by default; override with the `SENT_TRADER_DB` environment variable). No server setup needed.

For exploration and the price-vs-sentiment plot, open [notebooks/sent_scanner.ipynb](notebooks/sent_scanner.ipynb).

## Historical data

Live scraping only sees today's headlines; models need years. `sent-trader backfill` streams the [FNSPID dataset](https://huggingface.co/datasets/Zihan1004/FNSPID) (Dong et al. 2024 — ~15M timestamped, ticker-tagged financial news records, 1999–2023) directly over HTTP, keeps only rows for the 50-ticker universe ([universe.py](src/sent_trader/ingestion/universe.py)), and stores them alongside full daily price history from yfinance. Nothing multi-gigabyte lands on disk, re-running deduplicates, and the freshly stored backlog is scored by whichever model you choose.

[features.py](src/sent_trader/features.py) turns scored articles into model-ready daily features (mean score, article count, dispersion) with the anti-leakage rule applied: a headline published after the 16:00 US/Eastern close counts toward the *next* trading day.

## Layout

```
src/sent_trader/
├── config.py        # all tunables in one place
├── db.py            # SQLite schema + queries (+ versioned migrations)
├── features.py      # daily sentiment features, leakage-safe date shift
├── ingestion/
│   ├── news.py      # Google News RSS scraper (raw titles)
│   ├── prices.py    # daily OHLCV via yfinance
│   ├── fnspid.py    # FNSPID historical news, streamed + filtered
│   └── universe.py  # the 50-ticker training universe
├── sentiment/
│   ├── vader.py         # VADER scorer — reference scorer interface
│   ├── phrasebank.py    # Financial PhraseBank download + stratified splits
│   ├── tf_classifier.py # from-scratch TensorFlow classifier
│   └── benchmark.py     # head-to-head evaluation on the held-out set
├── pipeline.py      # scrape → store → score-what's-unscored
└── cli.py           # sent-trader command
tests/               # pytest suite (run: pytest)
```

## Schema

- `stock_list(stock_id, symbol)`
- `articles(article_id, link, title RAW, publish_date, stock_id)` — UNIQUE per (link, stock): one roundup article can cover several tickers
- `sentiment_scores(article_id, model_name, model_version, score, label, scored_at)` — UNIQUE per (article, model, version)
- `daily_stock_price(stock_id, date, open, high, low, close, volume)` — UNIQUE per (date, stock)

All inserts deduplicate (`ON CONFLICT DO NOTHING`), so re-running a scan is always safe.

## Sentiment models

**VADER** is the fast, dependency-light baseline. It scores raw headlines — intensifiers, punctuation and capitalization all carry signal for it, which is why titles are never stemmed.

**The TensorFlow classifier** is trained from scratch (no pretrained weights) on [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al. 2014, ~4.8k hand-labeled financial sentences, 50%-agreement variant). Because the training set is small, the model is deliberately small too: text vectorization → 64-d embeddings → average pooling → dense, with dropout, class weights and early stopping. Its output score is P(positive) − P(negative), on the same [-1, 1] scale as VADER's compound score.

Benchmark on the held-out PhraseBank test set (727 sentences, stratified, seed 42):

| model | accuracy | macro F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| VADER (nltk-3.10.0) | 0.545 | 0.490 | 0.348 | 0.616 | 0.506 |
| **tf-phrasebank (0.1.0)** | **0.726** | **0.686** | **0.642** | **0.805** | **0.611** |

Macro F1 is the headline metric: the dataset is ~59% neutral, so accuracy alone rewards predicting "neutral" everywhere. The biggest win is on negative headlines — the class that matters most for price reaction.

## Probabilistic forecaster

`sent-trader forecast NVDA` answers: *given the last month of prices and news sentiment, what does tomorrow's return distribution look like?*

The model ([forecast/model.py](src/sent_trader/forecast/model.py)) is a small quantile-regression MLP (7 quantiles, pinball loss, a non-crossing output layer) over 25 features per day: 20 vol-scaled return lags, trailing volatility, and 4 sentiment features from the TF classifier's scores. It predicts quantiles of the **vol-standardized** return and rescales by trailing volatility — one pooled network then serves calm and turbulent regimes alike. Trained on ~134k samples pooled across the 50-ticker universe (2009–2019), tuned on 2020–2021, tested on 2022–2023. Splits are chronological; a leakage test pins the target to strictly-next-day returns.

The bar it must clear ([forecast/baseline.py](src/sent_trader/forecast/baseline.py)): trailing empirical quantiles — "tomorrow will be distributed like the recent past." On the held-out 2022–2023 test set (25,050 ticker-days):

| model | pinball ↓ | 90% coverage (→0.90) | 50% coverage (→0.50) | 90% width |
|---|---|---|---|---|
| baseline 20d | 0.00446 | 0.818 | 0.458 | 0.0517 |
| baseline 250d | 0.00434 | **0.890** | 0.507 | 0.0601 |
| **tf-quantile** | **0.00432** | 0.859 | 0.469 | 0.0537 |

Honest read: the model wins on pinball loss (the proper scoring rule) with narrower intervals than the 250-day baseline, but its 90% band slightly undercovers (0.86 vs the target 0.90) — the baseline is better calibrated at the far tails. Daily equity returns are mostly noise; beating a well-chosen empirical baseline at all is the hard part, and the margin is accordingly thin.

## Roadmap

1. ✅ **Professional structure** — packaged layout, versioned scores, raw storage, tests, CLI
2. ✅ **TensorFlow sentiment classifier** trained on Financial PhraseBank — beats VADER on the held-out set (macro F1 0.686 vs 0.490)
3. ✅ **Historical data at scale** — FNSPID ingestion for the 50 largest S&P 500 names, streamed and batch-scored
4. ✅ **Probabilistic return forecaster** — quantile model over last-month prices + sentiment, beats the empirical-quantile baseline on pinball loss (0.00432 vs 0.00434 on 2022–2023)
5. **Streamlit app** — ticker picker, probability curve, headlines, calibration tab
