"""SQLite storage layer.

Design rules:
- Articles are stored with their RAW title. No preprocessing before storage,
  so any future model can re-score history from the original text.
- Sentiment lives in its own table, keyed by (article, model, version).
  Swapping VADER for a trained classifier later means adding rows, not
  rewriting them.
"""

import sqlite3
from datetime import datetime

import pandas as pd

from .config import config


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_list (
            stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS articles (
            article_id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            publish_date TEXT NOT NULL,
            stock_id INTEGER NOT NULL,
            FOREIGN KEY (stock_id) REFERENCES stock_list(stock_id)
        );

        CREATE TABLE IF NOT EXISTS sentiment_scores (
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

        CREATE TABLE IF NOT EXISTS daily_stock_price (
            dsp_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            date TEXT NOT NULL,
            UNIQUE (date, stock_id),
            FOREIGN KEY (stock_id) REFERENCES stock_list(stock_id)
        );
    """)
    conn.commit()
    conn.close()


def add_stock(symbol: str, db_path: str | None = None) -> int:
    """Insert the symbol if new and return its stock_id either way."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO stock_list (symbol) VALUES (?);", (symbol,))
    cur.execute("SELECT stock_id FROM stock_list WHERE symbol = ?;", (symbol,))
    stock_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return stock_id


def add_articles(stock_id: int, articles: list[dict], db_path: str | None = None) -> int:
    """Store raw articles, skipping links already present. Returns rows added."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    added = 0
    for art in articles:
        cur.execute(
            """
            INSERT INTO articles (link, title, publish_date, stock_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (link) DO NOTHING;
            """,
            (art["link"], art["title"], art["publish_date"].isoformat(), stock_id),
        )
        added += cur.rowcount
    conn.commit()
    conn.close()
    return added


def add_daily_prices(stock_id: int, price_data: pd.DataFrame, db_path: str | None = None) -> int:
    """Store daily OHLCV rows, skipping (date, stock) pairs already present."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    added = 0
    for _, row in price_data.iterrows():
        cur.execute(
            """
            INSERT INTO daily_stock_price (stock_id, open, high, low, close, volume, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date, stock_id) DO NOTHING;
            """,
            (
                stock_id,
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                int(row["Volume"]),
                row["Date"].isoformat(),
            ),
        )
        added += cur.rowcount
    conn.commit()
    conn.close()
    return added


def get_unscored_articles(model_name: str, model_version: str, db_path: str | None = None) -> pd.DataFrame:
    """Articles that the given model version has not scored yet."""
    conn = get_connection(db_path)
    df = pd.read_sql(
        """
        SELECT a.article_id, a.title
        FROM articles a
        LEFT JOIN sentiment_scores s
            ON s.article_id = a.article_id
            AND s.model_name = ?
            AND s.model_version = ?
        WHERE s.score_id IS NULL;
        """,
        conn,
        params=(model_name, model_version),
    )
    conn.close()
    return df


def add_sentiment_scores(scores: list[dict], db_path: str | None = None) -> int:
    """Store scores: dicts with article_id, model_name, model_version, score, label."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    added = 0
    for s in scores:
        cur.execute(
            """
            INSERT INTO sentiment_scores (article_id, model_name, model_version, score, label, scored_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (article_id, model_name, model_version) DO NOTHING;
            """,
            (s["article_id"], s["model_name"], s["model_version"], s["score"], s["label"], now),
        )
        added += cur.rowcount
    conn.commit()
    conn.close()
    return added


def get_stock_list(db_path: str | None = None) -> list[str]:
    conn = get_connection(db_path)
    df = pd.read_sql("SELECT symbol FROM stock_list ORDER BY symbol ASC;", conn)
    conn.close()
    return df["symbol"].tolist()


def get_stock_prices(ticker: str, db_path: str | None = None) -> pd.DataFrame:
    conn = get_connection(db_path)
    df = pd.read_sql(
        """
        SELECT p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_stock_price p
        JOIN stock_list s ON p.stock_id = s.stock_id
        WHERE s.symbol = ?
        ORDER BY p.date ASC;
        """,
        conn,
        params=(ticker,),
    )
    conn.close()
    return df


def get_article_sentiment(ticker: str, model_name: str = "vader", db_path: str | None = None) -> pd.DataFrame:
    """Raw titles with the given model's score; unscored articles have NULL score."""
    conn = get_connection(db_path)
    df = pd.read_sql(
        """
        SELECT a.publish_date, a.title, sc.score, sc.label, sc.model_version
        FROM articles a
        JOIN stock_list s ON s.stock_id = a.stock_id
        LEFT JOIN sentiment_scores sc
            ON sc.article_id = a.article_id AND sc.model_name = ?
        WHERE s.symbol = ?
        ORDER BY a.publish_date ASC;
        """,
        conn,
        params=(model_name, ticker),
    )
    conn.close()
    return df


def export_dataframes(db_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full price and article tables joined with symbols, for CSV export."""
    conn = get_connection(db_path)
    stock_df = pd.read_sql(
        """
        SELECT s.symbol, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_stock_price p
        JOIN stock_list s ON s.stock_id = p.stock_id;
        """,
        conn,
    )
    articles_df = pd.read_sql(
        """
        SELECT s.symbol, a.link, a.title, a.publish_date,
               sc.model_name, sc.model_version, sc.score, sc.label
        FROM articles a
        JOIN stock_list s ON s.stock_id = a.stock_id
        LEFT JOIN sentiment_scores sc ON sc.article_id = a.article_id;
        """,
        conn,
    )
    conn.close()
    return stock_df, articles_df
