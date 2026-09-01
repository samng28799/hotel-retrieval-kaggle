"""Tests for the Kaggle Hotel-ID adapter — both the flat (csv + images/) and
the folder-per-hotel layouts, plus the self-made query/gallery split.
"""

from pathlib import Path

import pandas as pd
import pytest

from hotelret import kaggle_data as kd


def _make_flat(root: Path, n_hotels=5, per=20):
    """michaln mirror layout: images/<id>.jpg + train.csv."""
    (root / "images").mkdir(parents=True)
    rows = []
    for h in range(n_hotels):
        for j in range(per):
            iid = f"img_{h}_{j}"
            (root / "images" / f"{iid}.jpg").write_bytes(b"\xff\xd8\xff")  # tiny jpeg stub
            rows.append({"image_id": iid, "hotel_id": f"hotel{h}"})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


def _make_foldered(root: Path, n_hotels=5, per=20):
    """competition layout: train_images/<hotel_id>/<id>.jpg, no csv."""
    for h in range(n_hotels):
        d = root / "train_images" / f"hotel{h}"
        d.mkdir(parents=True)
        for j in range(per):
            (d / f"img_{h}_{j}.jpg").write_bytes(b"\xff\xd8\xff")


def test_load_flat_layout(tmp_path):
    _make_flat(tmp_path)
    idx = kd.load_index(tmp_path)
    assert len(idx) == 100
    assert set(idx.columns) == {"image_id", "hotel_id", "path"}
    assert idx.hotel_id.nunique() == 5
    assert all(Path(p).exists() for p in idx.path)


def test_load_foldered_layout(tmp_path):
    _make_foldered(tmp_path)
    idx = kd.load_index(tmp_path)
    assert len(idx) == 100
    assert idx.hotel_id.nunique() == 5           # inferred from folder names
    assert "hotel0" in set(idx.hotel_id)


def test_summarize(tmp_path):
    _make_flat(tmp_path, n_hotels=4, per=20)
    s = kd.summarize(tmp_path)
    assert s["images"] == 80
    assert s["hotels"] == 4
    assert s["images_per_hotel_median"] == 20


def test_build_split_is_disjoint(tmp_path):
    _make_flat(tmp_path, n_hotels=5, per=20)
    gallery, query, unseen = kd.build_split(
        tmp_path, n_hotels=5, cap=10, min_images=15, unseen_frac=0.2, seed=0)
    # every hotel contributes exactly one query, rest capped into gallery
    assert query.hotel_id.nunique() == 5
    assert len(query) == 5
    assert (gallery.groupby("hotel_id").size() <= 10).all()
    # query images never leak into the gallery
    assert set(query.image_id).isdisjoint(set(gallery.image_id))
    assert len(unseen) == 1
    assert set(unseen).issubset(set(gallery.hotel_id))


def test_write_split(tmp_path):
    _make_flat(tmp_path, n_hotels=5, per=20)
    gallery, query, unseen = kd.build_split(tmp_path, n_hotels=5, cap=10, min_images=15)
    out = tmp_path / "manifest"
    summary = kd.write_split(gallery, query, unseen, out)
    assert (out / "gallery.csv").exists()
    assert (out / "queries.csv").exists()
    assert (out / "unseen_hotels.csv").exists()
    assert summary["hotels"] == 5
