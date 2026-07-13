from datetime import datetime

import pandas as pd
import pytest

from sent_trader import db, features


def test_effective_date_shifts_after_close():
    # January: US/Eastern is UTC-5, market closes 16:00 ET = 21:00 UTC
    dates = pd.Series(
        [
            "2023-01-10 20:59:00+00:00",  # 15:59 ET -> same day
            "2023-01-10 21:01:00+00:00",  # 16:01 ET -> next day
            "2023-01-10 03:00:00+00:00",  # 22:00 ET Jan 9 -> Jan 10
        ]
    )
    result = features.effective_date(dates).tolist()
    assert [d.isoformat() for d in result] == ["2023-01-10", "2023-01-11", "2023-01-10"]


def test_daily_sentiment_aggregates(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    stock_id = db.add_stock("NVDA", db_path)
    db.add_articles(
        stock_id,
        [
            # both 14:00 UTC = 09:00 ET, same trading day
            {"link": "https://e.com/1", "title": "up", "publish_date": datetime(2023, 1, 10, 14, 0)},
            {"link": "https://e.com/2", "title": "down", "publish_date": datetime(2023, 1, 10, 15, 0)},
        ],
        db_path,
    )
    pending = db.get_unscored_articles("m", "1", db_path)
    db.add_sentiment_scores(
        [
            {"article_id": int(a), "model_name": "m", "model_version": "1", "score": s, "label": "x"}
            for a, s in zip(pending["article_id"], [0.8, -0.2])
        ],
        db_path,
    )

    daily = features.daily_sentiment("NVDA", model_name="m", db_path=db_path)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["article_count"] == 2
    assert row["mean_score"] == pytest.approx(0.3)
    assert row["score_std"] > 0
