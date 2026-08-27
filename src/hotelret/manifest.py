"""Build the working subset and the unseen-hotel evaluation split.

The official Hotels-50K test protocol has ZERO unseen hotels — every test
hotel is also in training. To measure generalization to new properties we
carve out our own held-out set here. This is the core methodological choice
of the project, so it lives in one auditable place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import data


def build(metadata_dir, n_hotels=1000, cap=40, min_train=15,
          unseen_frac=0.20, seed=0):
    """Return (gallery_df, query_df, unseen_hotels list).

    gallery: training images for the selected hotels, capped per hotel.
    query:   official TraffickCam test images for those same hotels.
    unseen:  the held-out hotel_ids whose gallery is removed at eval time to
             simulate a never-before-seen property.
    """
    train = data.load_train(metadata_dir)
    test = data.load_test(metadata_dir)
    per = data.images_per_hotel(train)

    # eligible = appears in the official test set AND has enough gallery images
    cand = [h for h in test.hotel_id.unique() if per.get(h, 0) >= min_train]
    selected = cand[:n_hotels]
    sel = set(selected)

    gallery = (train[train.hotel_id.isin(sel)]
               .groupby("hotel_id").head(cap)
               .reset_index(drop=True))
    query = test[test.hotel_id.isin(sel)].reset_index(drop=True)

    # deterministic held-out split
    rng = pd.Series(selected).sample(frac=1.0, random_state=seed).tolist()
    n_unseen = max(1, int(len(rng) * unseen_frac))
    unseen = sorted(rng[:n_unseen])

    return gallery, query, unseen


def write(gallery, query, unseen, out_dir="data/manifest"):
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


def main():
    ap = argparse.ArgumentParser(description="Build subset + unseen-hotel split")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--hotels", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--min-train", type=int, default=15)
    ap.add_argument("--unseen-frac", type=float, default=0.20)
    ap.add_argument("--out", default="data/manifest")
    args = ap.parse_args()

    gallery, query, unseen = build(
        args.metadata, args.hotels, args.cap, args.min_train, args.unseen_frac)
    summary = write(gallery, query, unseen, args.out)
    print(json.dumps(summary, indent=2))
    print(f"wrote manifests to {args.out}/")


if __name__ == "__main__":
    main()
