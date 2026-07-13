import io
from datetime import datetime, timezone

import pytest

from sent_trader import db
from sent_trader.ingestion import fnspid

CSV_FIXTURE = """\
Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary
2020-06-05 06:30:54 UTC,NVDA hits new high,NVDA,https://example.com/nvda-high,Pub,,,,,,
2020-06-05 07:00:00 UTC,"Movers, and shakers roundup",NVDA,https://example.com/roundup,Pub,,,,,,
2020-06-05 07:00:00 UTC,"Movers, and shakers roundup",AAPL,https://example.com/roundup,Pub,,,,,,
2020-06-05 08:00:00 UTC,Some unrelated ticker,ZZZZ,https://example.com/zzz,Pub,,,,,,
bad-date,Broken row,NVDA,https://example.com/broken,Pub,,,,,,
"""


class FakeResponse:
    def __init__(self, data: bytes):
        self.raw = io.BytesIO(data)
        self.headers = {"content-length": str(len(data))}

    def raise_for_status(self):
        pass

    def close(self):
        pass


def test_parse_date_utc():
    parsed = fnspid.parse_date("2020-06-05 06:30:54 UTC")
    assert parsed == datetime(2020, 6, 5, 6, 30, 54, tzinfo=timezone.utc)


def test_ingest_filters_and_dedupes(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(
        fnspid.requests, "get", lambda *a, **k: FakeResponse(CSV_FIXTURE.encode())
    )

    result = fnspid.ingest_news(["NVDA", "AAPL"], db_path, log=lambda *_: None)
    assert result["rows_read"] == 5
    # ZZZZ filtered out, bad date skipped; same URL stored for BOTH tickers
    assert result["rows_matched"] == 3
    assert result["articles_added"] == 3

    nvda = db.get_article_sentiment("NVDA", db_path=db_path)
    aapl = db.get_article_sentiment("AAPL", db_path=db_path)
    assert len(nvda) == 2
    assert len(aapl) == 1
    assert aapl["title"].iloc[0] == "Movers, and shakers roundup"

    # Re-running adds nothing (dedup on link+stock)
    monkeypatch.setattr(
        fnspid.requests, "get", lambda *a, **k: FakeResponse(CSV_FIXTURE.encode())
    )
    rerun = fnspid.ingest_news(["NVDA", "AAPL"], db_path, log=lambda *_: None)
    assert rerun["articles_added"] == 0


def test_v1_schema_migrates_to_link_per_stock(tmp_path):
    import sqlite3

    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE stock_list (
            stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE
        );
        CREATE TABLE articles (
            article_id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            publish_date TEXT NOT NULL,
            stock_id INTEGER NOT NULL,
            FOREIGN KEY (stock_id) REFERENCES stock_list(stock_id)
        );
        CREATE TABLE sentiment_scores (
            score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            score REAL NOT NULL,
            label TEXT NOT NULL,
            scored_at TEXT NOT NULL,
            UNIQUE (article_id, model_name, model_version),
            FOREIGN KEY (article_id) REFERENCES articles(article_id)
        );
        INSERT INTO stock_list (symbol) VALUES ('NVDA'), ('AAPL');
        INSERT INTO articles (link, title, publish_date, stock_id)
        VALUES ('https://example.com/x', 'Old row', '2020-01-01T00:00:00', 1);
        INSERT INTO sentiment_scores (article_id, model_name, model_version, score, label, scored_at)
        VALUES (1, 'vader', 'v1', 0.5, 'positive', '2020-01-02T00:00:00');
    """)
    conn.commit()
    conn.close()

    db.init_db(db_path)  # runs the migration

    # Old row survived, and the same link can now exist for another stock
    added = db.add_articles(
        2,
        [{"link": "https://example.com/x", "title": "Old row", "publish_date": datetime(2020, 1, 1)}],
        db_path,
    )
    assert added == 1
    assert len(db.get_article_sentiment("NVDA", db_path=db_path)) == 1
    assert len(db.get_article_sentiment("AAPL", db_path=db_path)) == 1
    # scores survived the table rebuild with their article_id intact
    nvda = db.get_article_sentiment("NVDA", db_path=db_path)
    assert nvda["score"].iloc[0] == 0.5
