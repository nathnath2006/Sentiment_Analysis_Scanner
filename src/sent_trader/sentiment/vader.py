"""VADER sentiment scorer.

Scores RAW headline text: VADER's lexicon depends on intensifiers,
capitalization and punctuation, so no stemming or tokenization is applied.

This module is the reference implementation of the scorer interface every
future model (e.g. the TensorFlow classifier of milestone 2) must follow:
a MODEL_NAME, a MODEL_VERSION, and score_titles(titles) -> list of
{score, label}.
"""

import os

import certifi
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

MODEL_NAME = "vader"
MODEL_VERSION = f"nltk-{nltk.__version__}"

# Standard VADER compound-score thresholds.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            # Point the downloader at certifi's CA bundle; stock macOS
            # Pythons otherwise fail TLS verification here.
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            nltk.download("vader_lexicon", quiet=True)
            _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def label_for(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def score_titles(titles: list[str]) -> list[dict]:
    """Score headlines; returns one {score, label} dict per title, in order."""
    analyzer = _get_analyzer()
    results = []
    for title in titles:
        compound = analyzer.polarity_scores(title)["compound"]
        results.append({"score": compound, "label": label_for(compound)})
    return results
