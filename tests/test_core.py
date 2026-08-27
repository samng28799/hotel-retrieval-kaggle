"""Tests for the pieces that need neither network nor GPU: the metadata
header trap, the manifest split, embedding normalization, and the top-K
retrieval metric on a hand-built toy gallery.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hotelret import data, manifest, embed, evaluate


@pytest.fixture
def fake_metadata(tmp_path):
    """Write minimal train/test/hotel CSVs matching the real quirks:
    train_set.csv has NO header; test_set.csv HAS one."""
    # 3 hotels, a few images each
    train_rows = []
    for hid in ["h1", "h2", "h3"]:
        for j in range(20):  # 20 imgs each -> passes min_train=15
            train_rows.append(f"img_{hid}_{j},{hid},http://x/{hid}_{j}.jpg,travel_website,2019")
    (tmp_path / "train_set.csv").write_text("\n".join(train_rows) + "\n")

    test_rows = ["image_id,hotel_id,image_url,image_source,upload_timestamp"]
    for hid in ["h1", "h2", "h3"]:
        test_rows.append(f"q_{hid},{hid},http://x/q_{hid}.jpg,traffickcam,2019")
    (tmp_path / "test_set.csv").write_text("\n".join(test_rows) + "\n")

    hotels = ["hotel_id,hotel_name,chain_id,latitude,longitude",
              "h1,Hotel One,c1,0,0",
              "h2,Hotel Two,-1,0,0",
              "h3,Hotel Three,c1,0,0"]
    (tmp_path / "hotel_info.csv").write_text("\n".join(hotels) + "\n")
    return tmp_path


def test_train_has_no_header(fake_metadata):
    """The first train row must survive — the classic header-row trap."""
    train = data.load_train(fake_metadata)
    assert len(train) == 60          # 3 hotels x 20, none dropped
    assert "img_h1_0" in set(train.image_id)
    assert list(train.columns) == data.TRAIN_COLS


def test_summary_counts(fake_metadata):
    s = data.summarize(fake_metadata)
    assert s["train_records"] == 60
    assert s["train_hotels"] == 3
    assert s["test_hotels"] == 3
    # all test hotels are in train -> the key finding, reproduced in miniature
    assert s["test_hotels_unseen_in_train"] == 0
    # chain known for h1,h3 (40 imgs) not h2 (20) -> 40/60
    assert s["chain_known_frac"] == pytest.approx(40 / 60, abs=1e-3)


def test_manifest_split_is_disjoint(fake_metadata):
    gallery, query, unseen = manifest.build(
        fake_metadata, n_hotels=3, cap=10, min_train=15, unseen_frac=0.34, seed=0)
    assert gallery.hotel_id.nunique() == 3
    assert (gallery.groupby("hotel_id").size() <= 10).all()   # cap respected
    assert len(unseen) == 1                                    # 34% of 3 -> 1
    assert set(unseen).issubset(set(gallery.hotel_id))


def test_l2_normalize():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype="float32")
    out = embed.l2_normalize(x)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms[[0, 2]], 1.0)     # non-zero rows unit length
    assert np.allclose(out[1], 0.0)            # zero row stays zero, no NaN


def test_topk_accuracy_toy():
    # gallery: two hotels, two vectors each, clearly separated
    gallery_vecs = embed.l2_normalize(np.array([
        [1.0, 0.0], [0.9, 0.1],     # hotel A
        [0.0, 1.0], [0.1, 0.9],     # hotel B
    ], dtype="float32"))
    gallery_hotels = ["A", "A", "B", "B"]

    # queries sit right next to their true hotel
    query_vecs = embed.l2_normalize(np.array([
        [0.95, 0.05],   # -> A
        [0.05, 0.95],   # -> B
    ], dtype="float32"))
    query_hotels = ["A", "B"]

    res = embed.l2_normalize  # noqa: keep import used
    out = evaluate.topk_accuracy(query_vecs, query_hotels,
                                 gallery_vecs, gallery_hotels, ks=(1,))
    assert out["top1"] == 1.0       # both retrieved correctly at rank 1


def test_topk_accuracy_all_wrong():
    gallery_vecs = embed.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], "float32"))
    gallery_hotels = ["A", "B"]
    # query nearest to A's vector but its true hotel is B -> miss at top1
    query_vecs = embed.l2_normalize(np.array([[1.0, 0.0]], "float32"))
    out = evaluate.topk_accuracy(query_vecs, ["B"], gallery_vecs, gallery_hotels,
                                 ks=(1,))
    assert out["top1"] == 0.0
