"""Sentiment scorers.

Every scorer module exposes the same interface:
MODEL_NAME, MODEL_VERSION, and score_titles(titles) -> [{score, label}].
"""


def get_scorer(name: str):
    if name == "vader":
        from . import vader

        return vader
    if name in ("tf", "tf-phrasebank"):
        from . import tf_classifier

        return tf_classifier
    raise ValueError(f"Unknown sentiment model {name!r}. Choose 'vader' or 'tf'.")
