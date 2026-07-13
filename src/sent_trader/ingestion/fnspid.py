"""FNSPID historical news ingestion.

FNSPID (Dong et al. 2024) provides ~15M timestamped, ticker-tagged financial
news records. We stream Stock_news/All_external.csv (~5.7 GB: headline,
timestamp, ticker, URL — no article bodies) over HTTP and keep only rows for
our ticker universe, so nothing large ever lands on disk.

Re-running is safe: inserts deduplicate on (link, stock_id). Interrupting is
safe for the same reason, but a re-run restarts the stream from the top.
"""

import csv
import io
from datetime import datetime, timezone

import requests

from .. import db

FNSPID_NEWS_URL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID"
    "/resolve/main/Stock_news/All_external.csv"
)

# All_external.csv columns
COL_DATE = "Date"
COL_TITLE = "Article_title"
COL_SYMBOL = "Stock_symbol"
COL_URL = "Url"

BATCH_SIZE = 5_000


def parse_date(raw: str) -> datetime:
    """Parse FNSPID timestamps, e.g. '2020-06-05 06:30:54 UTC'."""
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=timezone.utc)


def _flush(batches: dict[str, list[dict]], stock_ids: dict[str, int], db_path: str | None) -> int:
    added = 0
    for symbol, rows in batches.items():
        if rows:
            added += db.add_articles(stock_ids[symbol], rows, db_path)
            rows.clear()
    return added


def ingest_news(
    tickers: list[str],
    db_path: str | None = None,
    max_rows: int | None = None,
    log=print,
) -> dict:
    """Stream the FNSPID news file and store rows matching `tickers`.

    max_rows caps the number of CSV rows read (for smoke tests).
    Returns counts: rows_read, rows_matched, articles_added.
    """
    db.init_db(db_path)
    wanted = set(tickers)
    stock_ids = {t: db.add_stock(t, db_path) for t in tickers}
    batches: dict[str, list[dict]] = {t: [] for t in tickers}

    response = requests.get(FNSPID_NEWS_URL, stream=True, timeout=60)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))

    text = io.TextIOWrapper(response.raw, encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(text)

    rows_read = matched = added = pending = 0
    try:
        for row in reader:
            rows_read += 1
            symbol = row.get(COL_SYMBOL)
            if symbol in wanted and row.get(COL_TITLE) and row.get(COL_URL):
                try:
                    publish_date = parse_date(row[COL_DATE])
                except (ValueError, TypeError):
                    continue
                batches[symbol].append(
                    {
                        "title": row[COL_TITLE].strip(),
                        "link": row[COL_URL].strip(),
                        "publish_date": publish_date,
                    }
                )
                matched += 1
                pending += 1

            if pending >= BATCH_SIZE:
                added += _flush(batches, stock_ids, db_path)
                pending = 0
                pct = 100 * response.raw.tell() / total_bytes if total_bytes else 0
                log(f"  {pct:5.1f}%  read {rows_read:,} rows, matched {matched:,}, stored {added:,}")

            if max_rows is not None and rows_read >= max_rows:
                break
    finally:
        added += _flush(batches, stock_ids, db_path)
        response.close()

    return {"rows_read": rows_read, "rows_matched": matched, "articles_added": added}
