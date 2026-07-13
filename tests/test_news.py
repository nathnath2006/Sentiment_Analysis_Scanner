from datetime import datetime, timezone

from sent_trader.ingestion.news import clean_title, parse_pubdate


def test_clean_title_strips_html():
    assert clean_title("<b>NVDA</b> hits record &amp; rallies") == "NVDA hits record & rallies"


def test_clean_title_plain_text_unchanged():
    assert clean_title("Plain headline") == "Plain headline"


def test_parse_pubdate_rfc822():
    parsed = parse_pubdate("Mon, 14 Jul 2025 13:05:00 GMT")
    assert parsed == datetime(2025, 7, 14, 13, 5, tzinfo=timezone.utc)


def test_parse_pubdate_missing_falls_back_to_now():
    parsed = parse_pubdate(None)
    assert (datetime.now(timezone.utc) - parsed).total_seconds() < 5
