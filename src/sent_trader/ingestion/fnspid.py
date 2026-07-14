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
import time
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


class ResilientHTTPStream(io.RawIOBase):
    """A byte stream over HTTP that survives dropped connections.

    A 5.7 GB download over a home connection WILL get reset at some point;
    this tracks the byte offset and transparently reconnects with an HTTP
    Range header, so the CSV parser above never notices. The retry budget
    refills on successful reads — only consecutive failures give up.
    """

    def __init__(self, url: str, max_retries: int = 10):
        self.url = url
        self.max_retries = max_retries
        self.pos = 0
        self.total = 0
        self._retries = 0
        self._connect(first=True)

    def _connect(self, first: bool = False) -> None:
        headers = {} if self.pos == 0 else {"Range": f"bytes={self.pos}-"}
        self._response = requests.get(self.url, stream=True, timeout=60, headers=headers)
        self._response.raise_for_status()
        if first:
            self.total = int(self._response.headers.get("content-length", 0))

    def _reconnect(self) -> None:
        self._retries += 1
        if self._retries > self.max_retries:
            raise ConnectionError(
                f"Stream failed {self.max_retries} consecutive times at byte {self.pos:,}"
            )
        time.sleep(min(2**self._retries, 30))
        try:
            self._connect()
        except requests.RequestException:
            self._reconnect()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while True:
            try:
                chunk = self._response.raw.read(len(buffer))
            except Exception:
                self._reconnect()
                continue
            if not chunk and self.total and self.pos < self.total:
                self._reconnect()  # silent truncation, not EOF
                continue
            self.pos += len(chunk)
            self._retries = 0
            buffer[: len(chunk)] = chunk
            return len(chunk)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            super().close()


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

    stream = ResilientHTTPStream(FNSPID_NEWS_URL)
    text = io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8", errors="replace", newline="")
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
                pct = 100 * stream.pos / stream.total if stream.total else 0
                log(f"  {pct:5.1f}%  read {rows_read:,} rows, matched {matched:,}, stored {added:,}")

            if max_rows is not None and rows_read >= max_rows:
                break
    finally:
        added += _flush(batches, stock_ids, db_path)
        text.close()

    return {"rows_read": rows_read, "rows_matched": matched, "articles_added": added}
