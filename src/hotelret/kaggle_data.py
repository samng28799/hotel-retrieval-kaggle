"""Kaggle Hotel-ID 2022 (FGVC9) adapter.

Drop-in replacement for the Hotels-50K data source after the feasibility audit
showed the Hotels-50K URLs decayed unevenly (67% survival, 15x overdispersion).
Kaggle ships decoded image files, so there is nothing to probe and nothing to
download image-by-image — the whole set arrives as one archive.

Layout after `kaggle competitions download` + unzip:
    <root>/train_images/<hotel_id>/<image_id>.jpg   (competition original)
or, using the pre-resized mirror michaln/hotelid-2022-train-images-256x256:
    <root>/images/<image_id>.jpg
    <root>/train.csv   with columns image_id, hotel_id

This module normalizes both layouts into the same (image_id, hotel_id, path)
frame the rest of the pipeline expects, so `embed` and `evaluate` are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_train_csv(root: Path) -> Path | None:
    for name in ("train.csv", "train_labels.csv"):
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


def load_index(root: str | Path) -> pd.DataFrame:
    """Return a DataFrame with columns image_id, hotel_id, path.

    Works with either the competition layout (folders per hotel) or the
    resized-mirror layout (flat images/ + train.csv).
    """
    root = Path(root)
    csv = _find_train_csv(root)

    if csv is not None:
        df = pd.read_csv(csv, dtype=str)
        # locate the actual image for each id (folders vary between mirrors)
        all_imgs = {p.stem: p for p in root.rglob("*.jpg")}
        df["path"] = df.image_id.map(lambda i: str(all_imgs.get(i, "")))
        df = df[df.path != ""].reset_index(drop=True)
        return df[["image_id", "hotel_id", "path"]]

    # no csv: infer hotel_id from the parent folder name
    rows = []
    for p in root.rglob("*.jpg"):
        rows.append({"image_id": p.stem, "hotel_id": p.parent.name, "path": str(p)})
    if not rows:
        raise FileNotFoundError(
            f"No train.csv and no .jpg files found under {root}. "
            "Did the download/unzip succeed?")
    return pd.DataFrame(rows)


def images_per_hotel(index: pd.DataFrame) -> pd.Series:
    return index.groupby("hotel_id").size().sort_values(ascending=False)


def summarize(root: str | Path) -> dict:
    idx = load_index(root)
    per = images_per_hotel(idx)
    return {
        "source": "kaggle-hotel-id-2022-fgvc9",
        "images": int(len(idx)),
        "hotels": int(idx.hotel_id.nunique()),
        "images_per_hotel_median": float(per.median()),
        "images_per_hotel_mean": round(float(per.mean()), 1),
        "images_per_hotel_min": int(per.min()),
        "images_per_hotel_max": int(per.max()),
        "hotels_with_ge5": int((per >= 5).sum()),
    }


def build_split(root, n_hotels=1000, cap=40, min_images=15,
                unseen_frac=0.20, seed=0):
    """Build gallery/query manifests and an unseen-hotel split.

    Kaggle has no separate labelled query set (test labels are hidden), so we
    make our own: for each selected hotel, hold out some images as queries and
    keep the rest as gallery. The unseen split additionally removes a subset of
    hotels' galleries entirely, exactly as with Hotels-50K.
    """
    idx = load_index(root)
    per = images_per_hotel(idx)

    eligible = per[per >= min_images].index.tolist()
    selected = eligible[:n_hotels]
    sub = idx[idx.hotel_id.isin(selected)].copy()

    # per hotel: last image (deterministic) -> query, rest (capped) -> gallery
    sub = sub.sort_values(["hotel_id", "image_id"]).reset_index(drop=True)
    query_rows, gallery_rows = [], []
    for hid, grp in sub.groupby("hotel_id"):
        grp = grp.reset_index(drop=True)
        query_rows.append(grp.iloc[-1])                 # 1 held-out query
        gallery_rows.append(grp.iloc[:-1].head(cap))    # rest as gallery, capped
    gallery = pd.concat(gallery_rows).reset_index(drop=True)
    query = pd.DataFrame(query_rows).reset_index(drop=True)

    rng = pd.Series(selected).sample(frac=1.0, random_state=seed).tolist()
    n_unseen = max(1, int(len(rng) * unseen_frac))
    unseen = sorted(rng[:n_unseen])

    return gallery, query, unseen


def write_split(gallery, query, unseen, out_dir="data/manifest"):
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gallery.to_csv(out / "gallery.csv", index=False)
    query.to_csv(out / "queries.csv", index=False)
    pd.Series(unseen, name="hotel_id").to_csv(out / "unseen_hotels.csv", index=False)
    summary = {
        "gallery_images": int(len(gallery)),
        "query_images": int(len(query)),
        "hotels": int(gallery.hotel_id.nunique()),
        "unseen_hotels": int(len(unseen)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
