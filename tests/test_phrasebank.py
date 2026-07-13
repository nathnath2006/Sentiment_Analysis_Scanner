import numpy as np
import pytest

from sent_trader.sentiment import get_scorer
from sent_trader.sentiment.phrasebank import parse_lines


def test_parse_lines_splits_on_last_at_sign():
    df = parse_lines(
        [
            "Profit rose to EUR 5.0 mn @ up from EUR 2.5 mn .@positive",
            "The company reported flat sales .@neutral",
            "",
            "Layoffs will affect 500 employees .@negative",
        ]
    )
    assert len(df) == 3
    assert df["label"].tolist() == ["positive", "neutral", "negative"]
    # '@' inside the sentence must not break parsing
    assert "EUR 5.0 mn @ up" in df["sentence"].iloc[0]


def test_parse_lines_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_lines(["Some sentence@bullish"])


def test_get_scorer_vader():
    scorer = get_scorer("vader")
    assert scorer.MODEL_NAME == "vader"


def test_get_scorer_unknown():
    with pytest.raises(ValueError):
        get_scorer("finbert")


def test_probs_to_results_score_and_label():
    tf_classifier = pytest.importorskip("sent_trader.sentiment.tf_classifier")
    probs = np.array(
        [
            [0.1, 0.2, 0.7],  # positive
            [0.6, 0.3, 0.1],  # negative
            [0.2, 0.6, 0.2],  # neutral
        ]
    )
    results = tf_classifier.probs_to_results(probs)
    assert [r["label"] for r in results] == ["positive", "negative", "neutral"]
    assert results[0]["score"] == pytest.approx(0.6)   # 0.7 - 0.1
    assert results[1]["score"] == pytest.approx(-0.5)  # 0.1 - 0.6
    assert results[2]["score"] == pytest.approx(0.0)
