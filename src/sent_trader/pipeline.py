"""End-to-end pipeline: scrape -> store raw -> score whatever is unscored.

Scoring runs over the database, not over freshly scraped rows, so adding a
new model version automatically back-fills history.
"""

from . import db
from .ingestion import fnspid
from .ingestion.news import scrape_news
from .ingestion.prices import fetch_daily_prices
from .ingestion.universe import TOP50
from .sentiment import get_scorer


def scan(ticker: str, db_path: str | None = None, model: str = "vader") -> dict:
    """Collect news and prices for a ticker, then score pending articles.

    Returns counts of what was added.
    """
    db.init_db(db_path)
    stock_id = db.add_stock(ticker, db_path)

    articles = scrape_news(ticker)
    new_articles = db.add_articles(stock_id, articles, db_path)

    prices = fetch_daily_prices(ticker)
    new_prices = db.add_daily_prices(stock_id, prices, db_path)

    new_scores = score_pending(db_path, model)

    return {
        "ticker": ticker,
        "articles_added": new_articles,
        "price_rows_added": new_prices,
        "articles_scored": new_scores,
    }


def score_pending(
    db_path: str | None = None,
    model: str = "vader",
    batch_size: int = 10_000,
    log=None,
) -> int:
    """Score every article the chosen model version has not seen yet.

    Works in batches so millions of backfilled articles don't have to fit
    in one model call.
    """
    scorer = get_scorer(model)
    total = 0
    while True:
        pending = db.get_unscored_articles(
            scorer.MODEL_NAME, scorer.MODEL_VERSION, db_path, limit=batch_size
        )
        if pending.empty:
            break
        results = scorer.score_titles(pending["title"].tolist())
        rows = [
            {
                "article_id": int(article_id),
                "model_name": scorer.MODEL_NAME,
                "model_version": scorer.MODEL_VERSION,
                "score": r["score"],
                "label": r["label"],
            }
            for article_id, r in zip(pending["article_id"], results)
        ]
        total += db.add_sentiment_scores(rows, db_path)
        if log:
            log(f"  scored {total:,} articles with {scorer.MODEL_NAME}")
        if len(pending) < batch_size:
            break
    return total


def backfill(
    tickers: list[str] | None = None,
    db_path: str | None = None,
    model: str | None = "tf",
    max_rows: int | None = None,
    skip_prices: bool = False,
    log=print,
) -> dict:
    """Load FNSPID historical news + full price history for the universe.

    Safe to re-run and to resume: every insert deduplicates.
    """
    tickers = tickers or TOP50

    log(f"Streaming FNSPID news for {len(tickers)} tickers...")
    news = fnspid.ingest_news(tickers, db_path, max_rows=max_rows, log=log)

    price_rows = 0
    if not skip_prices:
        log("Fetching full daily price history via yfinance...")
        for ticker in tickers:
            stock_id = db.add_stock(ticker, db_path)
            prices = fetch_daily_prices(ticker, period="max")
            price_rows += db.add_daily_prices(stock_id, prices, db_path)
        log(f"  stored {price_rows:,} price rows")

    scored = 0
    if model:
        log(f"Scoring backlog with model '{model}'...")
        scored = score_pending(db_path, model, log=log)

    return {**news, "price_rows_added": price_rows, "articles_scored": scored}
