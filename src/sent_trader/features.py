"""Daily sentiment features per ticker — the model-facing view of the news.

The leakage rule lives here: a headline published after the 16:00 US/Eastern
close cannot influence that day's trading, so it belongs to the NEXT
calendar day's features. Weekends and holidays are resolved later, when
features are merged onto actual trading days (merge_asof forward).
"""

import pandas as pd

from . import db

MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16


def effective_date(publish_dates: pd.Series) -> pd.Series:
    """Map UTC publish timestamps to the trading date they can influence."""
    ts = pd.to_datetime(publish_dates, utc=True, format="mixed").dt.tz_convert(MARKET_TZ)
    after_close = ts.dt.hour >= MARKET_CLOSE_HOUR
    return (ts + after_close * pd.Timedelta(days=1)).dt.date


def daily_sentiment(ticker: str, model_name: str = "vader", db_path: str | None = None) -> pd.DataFrame:
    """Aggregate scored articles into one row per effective trading date.

    Columns: date, mean_score, article_count, score_std (0 for single-article
    days).
    """
    articles = db.get_article_sentiment(ticker, model_name, db_path)
    scored = articles.dropna(subset=["score"]).copy()
    if scored.empty:
        return pd.DataFrame(columns=["date", "mean_score", "article_count", "score_std"])

    scored["date"] = effective_date(scored["publish_date"])
    daily = (
        scored.groupby("date")["score"]
        .agg(mean_score="mean", article_count="count", score_std="std")
        .reset_index()
    )
    daily["score_std"] = daily["score_std"].fillna(0.0)
    return daily
