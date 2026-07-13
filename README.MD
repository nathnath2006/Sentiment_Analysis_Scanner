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
pip install -e ".[dev]"

sent-trader scan NVDA AAPL   # scrape news + prices, score sentiment
sent-trader list             # tickers in the database
sent-trader export           # dump prices and articles to CSV
```

The database is a single SQLite file (`sentiment.db` by default; override with the `SENT_TRADER_DB` environment variable). No server setup needed.

For exploration and the price-vs-sentiment plot, open [notebooks/sent_scanner.ipynb](notebooks/sent_scanner.ipynb).

## Layout

```
src/sent_trader/
├── config.py        # all tunables in one place
├── db.py            # SQLite schema + queries
├── ingestion/
│   ├── news.py      # Google News RSS scraper (raw titles)
│   └── prices.py    # daily OHLCV via yfinance
├── sentiment/
│   └── vader.py     # VADER scorer — reference scorer interface
├── pipeline.py      # scrape → store → score-what's-unscored
└── cli.py           # sent-trader command
tests/               # pytest suite (run: pytest)
```

## Schema

- `stock_list(stock_id, symbol)`
- `articles(article_id, link UNIQUE, title RAW, publish_date, stock_id)`
- `sentiment_scores(article_id, model_name, model_version, score, label, scored_at)` — UNIQUE per (article, model, version)
- `daily_stock_price(stock_id, date, open, high, low, close, volume)` — UNIQUE per (date, stock)

All inserts deduplicate (`ON CONFLICT DO NOTHING`), so re-running a scan is always safe.

## Why VADER (for now)

VADER is the fast, dependency-light baseline. It scores raw headlines — intensifiers, punctuation and capitalization all carry signal for it, which is why titles are never stemmed. It is also the bar the next model has to clear.

## Roadmap

1. ✅ **Professional structure** — packaged layout, versioned scores, raw storage, tests, CLI
2. **TensorFlow sentiment classifier** trained on Financial PhraseBank — must beat VADER on a held-out set
3. **Historical data at scale** — FNSPID ingestion for the 50 largest S&P 500 names
4. **Probabilistic return forecaster** — quantile model over last-month prices + sentiment, validated with a coverage/calibration test against an empirical-quantile baseline
5. **Streamlit app** — ticker picker, probability curve, headlines, calibration tab
