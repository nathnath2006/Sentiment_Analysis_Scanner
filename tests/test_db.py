from datetime import datetime

import pandas as pd
import pytest

from sent_trader import db


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


def make_articles(n=2):
    return [
        {
            "title": f"Headline {i}",
            "link": f"https://example.com/a{i}",
            "publish_date": datetime(2026, 7, 14, 12, i),
        }
        for i in range(n)
    ]


def make_prices():
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-07-13"), pd.Timestamp("2026-07-14")],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_100_000],
        }
    )


def test_add_stock_is_idempotent(db_path):
    first = db.add_stock("NVDA", db_path)
    second = db.add_stock("NVDA", db_path)
    assert first == second
    assert db.get_stock_list(db_path) == ["NVDA"]


def test_articles_deduplicate_by_link(db_path):
    stock_id = db.add_stock("NVDA", db_path)
    articles = make_articles()
    assert db.add_articles(stock_id, articles, db_path) == 2
    assert db.add_articles(stock_id, articles, db_path) == 0


def test_prices_deduplicate_by_date(db_path):
    stock_id = db.add_stock("NVDA", db_path)
    prices = make_prices()
    assert db.add_daily_prices(stock_id, prices, db_path) == 2
    assert db.add_daily_prices(stock_id, prices, db_path) == 0
    stored = db.get_stock_prices("NVDA", db_path)
    assert len(stored) == 2
    assert stored["close"].tolist() == [101.0, 102.0]


def test_sentiment_scores_unique_per_model_version(db_path):
    stock_id = db.add_stock("NVDA", db_path)
    db.add_articles(stock_id, make_articles(1), db_path)
    pending = db.get_unscored_articles("vader", "v1", db_path)
    assert len(pending) == 1

    score = {
        "article_id": int(pending["article_id"].iloc[0]),
        "model_name": "vader",
        "model_version": "v1",
        "score": 0.5,
        "label": "positive",
    }
    assert db.add_sentiment_scores([score], db_path) == 1
    assert db.add_sentiment_scores([score], db_path) == 0

    # Same article is "unscored" again for a new model version: re-scoring
    # history with a better model must always be possible.
    assert len(db.get_unscored_articles("vader", "v2", db_path)) == 1
    assert len(db.get_unscored_articles("vader", "v1", db_path)) == 0


def test_export_includes_scores(db_path):
    stock_id = db.add_stock("NVDA", db_path)
    db.add_articles(stock_id, make_articles(1), db_path)
    db.add_daily_prices(stock_id, make_prices(), db_path)
    pending = db.get_unscored_articles("vader", "v1", db_path)
    db.add_sentiment_scores(
        [
            {
                "article_id": int(pending["article_id"].iloc[0]),
                "model_name": "vader",
                "model_version": "v1",
                "score": -0.3,
                "label": "negative",
            }
        ],
        db_path,
    )
    stock_df, articles_df = db.export_dataframes(db_path)
    assert len(stock_df) == 2
    assert len(articles_df) == 1
    assert articles_df["label"].iloc[0] == "negative"
