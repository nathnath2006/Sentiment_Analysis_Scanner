"""End-to-end pipeline: scrape -> store raw -> score whatever is unscored.

Scoring runs over the database, not over freshly scraped rows, so adding a
new model version automatically back-fills history.
"""

from . import db
from .ingestion.news import scrape_news
from .ingestion.prices import fetch_daily_prices
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


def score_pending(db_path: str | None = None, model: str = "vader") -> int:
    """Score every article the chosen model version has not seen yet."""
    scorer = get_scorer(model)
    pending = db.get_unscored_articles(scorer.MODEL_NAME, scorer.MODEL_VERSION, db_path)
    if pending.empty:
        return 0
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
    return db.add_sentiment_scores(rows, db_path)
