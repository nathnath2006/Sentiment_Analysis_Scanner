"""News ingestion from Google News RSS.

Titles are cleaned of HTML but otherwise left untouched: no stemming, no
tokenization. Raw text goes to the database; models decide their own
preprocessing at scoring time.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests
from bs4 import BeautifulSoup

from ..config import config


def clean_title(title: str) -> str:
    """Strip any HTML markup from a headline."""
    return BeautifulSoup(title, "html.parser").get_text().strip()


def parse_pubdate(raw: str | None) -> datetime:
    """Parse an RSS pubDate (RFC 822, e.g. 'Mon, 14 Jul 2025 13:05:00 GMT').

    Falls back to the current UTC time when the feed omits the date.
    """
    if not raw:
        return datetime.now(timezone.utc)
    return parsedate_to_datetime(raw)


def scrape_news(ticker: str) -> list[dict]:
    """Fetch current headlines for a ticker.

    Returns a list of dicts: title (clean text), link, publish_date (datetime),
    ticker.
    """
    # Fetch with requests (certifi CA bundle) rather than feedparser's own
    # urllib fetch, which fails TLS verification on stock macOS Pythons.
    response = requests.get(config.news_feed_url.format(ticker=ticker), timeout=15)
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    articles = []
    for entry in feed.entries:
        articles.append(
            {
                "title": clean_title(entry.title),
                "link": entry.link,
                "publish_date": parse_pubdate(entry.get("published")),
                "ticker": ticker,
            }
        )
    return articles
