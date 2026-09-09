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


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp")


def _all_images(root: Path):
    """All image files under root, across common extensions."""
    out = []
    for ext in IMAGE_EXTS:
        out.extend(root.rglob(ext))
    return out


def load_index(root: str | Path) -> pd.DataFrame:
    """Return a DataFrame with columns image_id, hotel_id, path.

    Works with either the competition layout (folders per hotel) or the
    resized-mirror layout (flat images/ + train.csv). Falls back gracefully:
    if a CSV exists but its image_ids don't line up with files on disk, it
    infers labels from the folder structure instead of returning nothing.
    """
    root = Path(root)
    imgs = _all_images(root)
    if not imgs:
        raise FileNotFoundError(
            f"No image files ({', '.join(IMAGE_EXTS)}) found under {root}. "
            "Did the download/unzip actually run? Run the diagnostic cell.")

    by_stem = {p.stem: p for p in imgs}
    csv = _find_train_csv(root)

    if csv is not None:
        df = pd.read_csv(csv, dtype=str)
        # tolerate an image_id column that already includes an extension
        df["_stem"] = df.image_id.map(lambda i: Path(str(i)).stem)
        df["path"] = df["_stem"].map(lambda s: str(by_stem.get(s, "")))
        df = df[df.path != ""]
        if len(df) > 0:
            out = df[["image_id", "hotel_id", "path"]].reset_index(drop=True)
            out["image_id"] = out.path.map(lambda p: Path(p).stem)  # keep ids == file stems
            return out
        # CSV present but nothing matched -> fall through to folder inference

    # infer hotel_id from the immediate parent folder name
    rows = [{"image_id": p.stem, "hotel_id": p.parent.name, "path": str(p)}
            for p in imgs]
    return pd.DataFrame(rows)


def images_per_hotel(index: pd.DataFrame) -> pd.Series:
    return index.groupby("hotel_id").size().sort_values(ascending=False)


def summarize(root: str | Path) -> dict:
    idx = load_index(root)
    if len(idx) == 0:
        raise ValueError(
            f"No images found under {root}. Check that the download step ran and "
            "unzipped, that DATA_ROOT points at the unzipped folder, and that the "
            "images are .jpg. Run the diagnostic cell to inspect the folder tree."
        )
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
                unseen_frac=0.20, queries_per_hotel=1, seed=0):
    """Build gallery/query manifests and an unseen-hotel split.

    Kaggle has no separate labelled query set (test labels are hidden), so we
    make our own: for each selected hotel, hold out `queries_per_hotel` images
    as queries and keep the rest (capped) as gallery. Holding out more than one
    query per hotel raises the query count and shrinks sampling noise in the
    evaluation, at the cost of a slightly smaller gallery per hotel.
    """
    idx = load_index(root)
    per = images_per_hotel(idx)

    # a hotel must have enough images to give up `queries_per_hotel` and still
    # leave at least one gallery image behind
    need = max(min_images, queries_per_hotel + 1)
    eligible = per[per >= need].index.tolist()
    selected = eligible[:n_hotels]
    sub = idx[idx.hotel_id.isin(selected)].copy()

    sub = sub.sort_values(["hotel_id", "image_id"]).reset_index(drop=True)
    query_rows, gallery_rows = [], []
    q = queries_per_hotel
    for hid, grp in sub.groupby("hotel_id"):
        grp = grp.reset_index(drop=True)
        query_rows.append(grp.iloc[-q:])            # last q images -> queries
        gallery_rows.append(grp.iloc[:-q].head(cap))  # rest -> gallery, capped
    gallery = pd.concat(gallery_rows).reset_index(drop=True)
    query = pd.concat(query_rows).reset_index(drop=True)

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