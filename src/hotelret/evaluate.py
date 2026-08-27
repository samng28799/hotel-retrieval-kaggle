"""Retrieval evaluation: top-K accuracy by hotel and by chain.

The whole point of the project lives in the ``split`` argument:

- ``seen``   : gallery keeps all hotels (the standard, too-easy benchmark).
- ``unseen`` : query hotels listed in unseen_hotels.csv have their gallery
               images removed, so the correct hotel is genuinely absent and
               the model must fail gracefully / retrieve a plausible neighbour.

Comparing the two numbers, per encoder, is the headline result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_embeddings(path):
    emb = np.load(path)
    ids = json.loads(Path(path).with_suffix(".ids.json").read_text())
    return emb, {img_id: i for i, img_id in enumerate(ids)}


def topk_accuracy(query_vecs, query_hotels, gallery_vecs, gallery_hotels,
                  ks=(1, 10, 100)):
    """Fraction of queries whose correct hotel appears in the top-K gallery hits.

    Vectors are assumed L2-normalized, so cosine similarity is a dot product.
    Uses a per-query argpartition to stay memory-light on modest galleries.
    """
    sims = query_vecs @ gallery_vecs.T          # (Q, G)
    gallery_hotels = np.asarray(gallery_hotels)
    maxk = max(ks)
    # top maxk gallery indices per query (unordered within the top-k is fine
    # for a hit test, but we sort for rank-sensitive metrics later)
    top = np.argpartition(-sims, kth=min(maxk, sims.shape[1] - 1), axis=1)[:, :maxk]
    row = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row, top], axis=1)
    top = top[row, order]                        # now sorted best-first

    hit_hotels = gallery_hotels[top]             # (Q, maxk)
    correct = hit_hotels == np.asarray(query_hotels)[:, None]

    out = {}
    for k in ks:
        out[f"top{k}"] = round(float(correct[:, :k].any(axis=1).mean()), 4)
    return out


def evaluate(embeddings_path, manifest_dir="data/manifest", split="seen"):
    emb, idx = _load_embeddings(embeddings_path)

    gallery = pd.read_csv(Path(manifest_dir) / "gallery.csv", dtype=str)
    queries = pd.read_csv(Path(manifest_dir) / "queries.csv", dtype=str)
    unseen = set(pd.read_csv(Path(manifest_dir) / "unseen_hotels.csv",
                             dtype=str).hotel_id)

    if split == "unseen":
        # remove the query hotels' own gallery entries -> property truly unseen
        gallery = gallery[~gallery.hotel_id.isin(unseen)]
        queries = queries[queries.hotel_id.isin(unseen)]
    elif split != "seen":
        raise ValueError("split must be 'seen' or 'unseen'")

    # keep only rows we actually have embeddings for
    gallery = gallery[gallery.image_id.isin(idx)]
    queries = queries[queries.image_id.isin(idx)]

    g_vecs = emb[[idx[i] for i in gallery.image_id]]
    q_vecs = emb[[idx[i] for i in queries.image_id]]

    result = topk_accuracy(q_vecs, queries.hotel_id.tolist(),
                           g_vecs, gallery.hotel_id.tolist())
    result["split"] = split
    result["n_query"] = int(len(queries))
    result["n_gallery"] = int(len(gallery))
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval accuracy")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--manifest", default="data/manifest")
    ap.add_argument("--split", default="seen", choices=["seen", "unseen"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = evaluate(args.embeddings, args.manifest, args.split)
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
