"""Head-to-head evaluation on the PhraseBank held-out test split.

The bar the TF classifier must clear: beat VADER on macro F1 (macro, not
accuracy, because the dataset is ~59% neutral and accuracy rewards
neutral-spamming).
"""

from sklearn.metrics import accuracy_score, f1_score

from .phrasebank import LABELS, splits


def evaluate_scorer(scorer, sentences: list[str], true_labels: list[str]) -> dict:
    predicted = [r["label"] for r in scorer.score_titles(sentences)]
    per_class = f1_score(true_labels, predicted, labels=LABELS, average=None)
    return {
        "model": f"{scorer.MODEL_NAME} ({scorer.MODEL_VERSION})",
        "accuracy": round(accuracy_score(true_labels, predicted), 4),
        "macro_f1": round(f1_score(true_labels, predicted, average="macro"), 4),
        **{f"f1_{label}": round(score, 4) for label, score in zip(LABELS, per_class)},
    }


def run() -> list[dict]:
    """Evaluate VADER and the TF classifier on the same test sentences."""
    from . import tf_classifier, vader

    test = splits()["test"]
    sentences = test["sentence"].tolist()
    true_labels = test["label"].tolist()

    results = [
        evaluate_scorer(vader, sentences, true_labels),
        evaluate_scorer(tf_classifier, sentences, true_labels),
    ]

    header = ["model", "accuracy", "macro_f1"] + [f"f1_{l}" for l in LABELS]
    widths = [max(len(h), max(len(str(r[h])) for r in results)) for h in header]
    print(f"\nPhraseBank held-out test set: {len(sentences)} sentences\n")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in results:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(header, widths)))
    print()
    return results
