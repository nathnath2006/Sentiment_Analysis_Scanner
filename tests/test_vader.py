import pytest

from sent_trader.sentiment import vader


@pytest.fixture(scope="module")
def analyzer_available():
    try:
        vader._get_analyzer()
    except Exception:  # lexicon download needs network on first run
        pytest.skip("VADER lexicon unavailable (no network?)")


def test_label_thresholds():
    assert vader.label_for(0.5) == "positive"
    assert vader.label_for(-0.5) == "negative"
    assert vader.label_for(0.0) == "neutral"


def test_scores_raw_text(analyzer_available):
    results = vader.score_titles(
        [
            "Company reports fantastic record profits, stock soars!",
            "Company collapses amid fraud scandal, investors devastated",
        ]
    )
    assert results[0]["score"] > 0
    assert results[0]["label"] == "positive"
    assert results[1]["score"] < 0
    assert results[1]["label"] == "negative"


def test_scorer_interface_contract():
    # Every future model (e.g. the TF classifier) must expose these.
    assert isinstance(vader.MODEL_NAME, str)
    assert isinstance(vader.MODEL_VERSION, str)
    assert callable(vader.score_titles)
