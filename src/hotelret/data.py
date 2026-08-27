"""Load Hotels-50K metadata.

The single most important thing in this module: ``train_set.csv`` has **no
header row** while ``test_set.csv`` **does**. Loading the train file with
``header=0`` silently drops the first image and shifts column names. Every read
goes through here so that trap is handled exactly once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRAIN_COLS = ["image_id", "hotel_id", "image_url", "image_source", "upload_timestamp"]


def load_train(metadata_dir: str | Path) -> pd.DataFrame:
    """Load the training manifest. Note: no header row in the source file."""
    path = Path(metadata_dir) / "train_set.csv"
    return pd.read_csv(path, names=TRAIN_COLS, header=None, dtype=str)


def load_test(metadata_dir: str | Path) -> pd.DataFrame:
    """Load the test manifest. This one *does* have a header row."""
    path = Path(metadata_dir) / "test_set.csv"
    return pd.read_csv(path, dtype=str)


def load_hotels(metadata_dir: str | Path) -> pd.DataFrame:
    """Load per-hotel metadata (hotel_id, chain_id, lat/lon)."""
    path = Path(metadata_dir) / "hotel_info.csv"
    return pd.read_csv(path, dtype=str)


def images_per_hotel(train: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by hotel_id giving image count, descending."""
    return train.groupby("hotel_id").size().sort_values(ascending=False)


def summarize(metadata_dir: str | Path) -> dict:
    """Quick sanity summary — used by the audit and by tests."""
    train = load_train(metadata_dir)
    test = load_test(metadata_dir)
    hotels = load_hotels(metadata_dir)
    per = images_per_hotel(train)

    seen = set(train.hotel_id)
    test_hotels = set(test.hotel_id)

    merged = train.merge(hotels[["hotel_id", "chain_id"]], on="hotel_id", how="left")
    known_chain = int((merged.chain_id != "-1").sum())

    return {
        "train_records": int(len(train)),
        "train_hotels": int(train.hotel_id.nunique()),
        "test_records": int(len(test)),
        "test_hotels": int(test.hotel_id.nunique()),
        "test_hotels_unseen_in_train": int(len(test_hotels - seen)),
        "images_per_hotel_median": float(per.median()),
        "images_per_hotel_mean": round(float(per.mean()), 1),
        "chain_known_images": known_chain,
        "chain_known_frac": round(known_chain / len(merged), 3),
        "source_counts": train.image_source.value_counts().to_dict(),
    }
