"""From-scratch TensorFlow sentiment classifier trained on Financial PhraseBank.

Deliberately small: ~5k labeled sentences cannot feed a deep model, so this
is a bag-of-embeddings network (vectorize -> embed -> average -> dense) with
dropout and early stopping. No pretrained weights anywhere.

Implements the scorer interface defined by sentiment/vader.py:
MODEL_NAME, MODEL_VERSION, score_titles(titles) -> [{score, label}].
The score is P(positive) - P(negative), which lands in [-1, 1] like VADER's
compound score, so downstream aggregation treats both models identically.
"""

from pathlib import Path

import numpy as np

from ..config import config
from .phrasebank import LABELS, splits

MODEL_NAME = "tf-phrasebank"
MODEL_VERSION = "0.1.0"

MAX_TOKENS = 10_000
SEQUENCE_LENGTH = 48
EMBEDDING_DIM = 64

_model = None


def model_path() -> Path:
    return Path(config.models_dir) / f"{MODEL_NAME}-{MODEL_VERSION}.keras"


def build_model(train_texts: list[str]):
    import keras
    from keras import layers

    vectorize = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorize.adapt(train_texts)

    model = keras.Sequential(
        [
            keras.Input(shape=(1,), dtype="string"),
            vectorize,
            layers.Embedding(MAX_TOKENS, EMBEDDING_DIM, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(LABELS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 50, verbose: int = 2) -> dict:
    """Train on PhraseBank train split, early-stop on val, save the model."""
    import keras

    keras.utils.set_random_seed(config.seed)

    data = splits()
    train_df, val_df = data["train"], data["val"]

    y_train = np.array([LABELS.index(l) for l in train_df["label"]])
    y_val = np.array([LABELS.index(l) for l in val_df["label"]])

    # PhraseBank is ~59% neutral; without class weights the model can coast
    # by rarely predicting "negative".
    counts = np.bincount(y_train, minlength=len(LABELS))
    class_weight = {i: len(y_train) / (len(LABELS) * c) for i, c in enumerate(counts)}

    model = build_model(train_df["sentence"].tolist())
    model.fit(
        np.array(train_df["sentence"]),
        y_train,
        validation_data=(np.array(val_df["sentence"]), y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=verbose,
    )

    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

    val_loss, val_acc = model.evaluate(
        np.array(val_df["sentence"]), y_val, verbose=0
    )
    global _model
    _model = model
    return {
        "model_path": str(path),
        "train_sentences": len(train_df),
        "val_accuracy": round(float(val_acc), 4),
        "val_loss": round(float(val_loss), 4),
    }


def _get_model():
    global _model
    if _model is None:
        import keras

        path = model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: sent-trader train"
            )
        _model = keras.models.load_model(path)
    return _model


def probs_to_results(probs: np.ndarray) -> list[dict]:
    """Map softmax rows to the scorer contract: score = P(pos) - P(neg)."""
    results = []
    for row in probs:
        score = float(row[LABELS.index("positive")] - row[LABELS.index("negative")])
        results.append({"score": score, "label": LABELS[int(np.argmax(row))]})
    return results


def score_titles(titles: list[str]) -> list[dict]:
    if not titles:
        return []
    import tensorflow as tf

    # Direct __call__ rather than model.predict(): predict()'s threaded
    # tf.data pipeline can deadlock off the main thread (e.g. Streamlit).
    # tf.constant, not np.array: numpy converts str lists to unicode dtype
    # ('<U...'), which Keras rejects as a string-model input.
    model = _get_model()
    probs = np.concatenate(
        [
            model(tf.constant(titles[i : i + 2048]), training=False).numpy()
            for i in range(0, len(titles), 2048)
        ]
    )
    return probs_to_results(probs)
