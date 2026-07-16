

"""Financial PhraseBank (Malo et al. 2014) loading and splitting.

The dataset is ~4.8k financial news sentences hand-labeled positive /
neutral / negative. We use the "50% annotator agreement" variant (the
largest). Downloaded once from the HuggingFace mirror and cached locally.

Splits are stratified and seeded: train() and benchmark() must see the
exact same held-out test set or the VADER comparison is meaningless.
"""

import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from ..config import config

PHRASEBANK_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank"
    "/resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
ZIP_NAME = "FinancialPhraseBank-v1.0.zip"
AGREEMENT_FILES = {
    "50": "Sentences_50Agree.txt",
    "66": "Sentences_66Agree.txt",
    "75": "Sentences_75Agree.txt",
    "all": "Sentences_AllAgree.txt",
}
LABELS = ["negative", "neutral", "positive"]


def _zip_path() -> Path:
    return Path(config.data_dir) / ZIP_NAME


def download(force: bool = False) -> Path:
    """Fetch the PhraseBank zip into the data dir (cached after first run)."""
    path = _zip_path()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(PHRASEBANK_URL, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def parse_lines(lines: list[str]) -> pd.DataFrame:
    """Parse 'sentence@label' lines into a DataFrame with sentence, label."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sentence, _, label = line.rpartition("@")
        label = label.strip().lower()
        if label not in LABELS:
            raise ValueError(f"Unexpected label {label!r} in line: {line[:80]}")
        rows.append({"sentence": sentence.strip(), "label": label})
    return pd.DataFrame(rows)


def load(agreement: str = "50") -> pd.DataFrame:
    """Load the PhraseBank sentences for a given annotator-agreement level."""
    path = download()
    member = f"FinancialPhraseBank-v1.0/{AGREEMENT_FILES[agreement]}"
    with zipfile.ZipFile(path) as zf:
        # The original distribution is Latin-1 encoded.
        text = zf.read(member).decode("latin-1")
    return parse_lines(text.splitlines())


def splits(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Deterministic stratified 70/15/15 train/val/test split."""
    if df is None:
        df = load()
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=config.seed
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=train_val["label"],
        random_state=config.seed,
    )
    return {"train": train, "val": val, "test": test}
