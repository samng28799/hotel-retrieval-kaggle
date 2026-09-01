"""Retrieval evaluation: top-K accuracy by hotel.

The 'unseen' split models the deployment reality of a barely-known property:
a hotel that has just entered the gallery with almost no photos. Rather than
removing held-out hotels from the gallery entirely (which makes retrieval
impossible by construction and forces accuracy to zero), we keep a SINGLE
gallery image per held-out hotel. The correct hotel is therefore still
findable, but represented by minimal evidence -- the retrieval analogue of
Feizi et al.'s 'truly unseen' hotels, and a direct match to the new-listing
scenario in the introduction. Comparing top-K on these sparse-gallery hotels
against the full-gallery hotels measures the accuracy drop and whether the
ranking of encoders is stable.
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
    """
    sims = query_vecs @ gallery_vecs.T          # (Q, G)
    gallery_hotels = np.asarray(gallery_hotels)
    maxk = min(max(ks), sims.shape[1])
    top = np.argpartition(-sims, kth=maxk - 1, axis=1)[:, :maxk]
    row = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row, top], axis=1)
    top = top[row, order]                        # sorted best-first

    hit_hotels = gallery_hotels[top]             # (Q, maxk)
    correct = hit_hotels == np.asarray(query_hotels)[:, None]

    out = {}
    for k in ks:
        out[f"top{k}"] = round(float(correct[:, :k].any(axis=1).mean()), 4)
    return out


def evaluate(embeddings_path, manifest_dir="data/manifest", split="seen",
             sparse_k=1, seed=0):
    """Evaluate retrieval accuracy.

    split='seen'   : full gallery, all queries.
    split='unseen' : held-out (20%) hotels keep only `sparse_k` gallery image(s);
                     queries are restricted to those held-out hotels. The hotel
                     stays findable, so accuracy is meaningful (not forced to 0).
    """
    emb, idx = _load_embeddings(embeddings_path)

    gallery = pd.read_csv(Path(manifest_dir) / "gallery.csv", dtype=str)
    queries = pd.read_csv(Path(manifest_dir) / "queries.csv", dtype=str)
    unseen = set(pd.read_csv(Path(manifest_dir) / "unseen_hotels.csv",
                             dtype=str).hotel_id)

    if split == "unseen":
        # keep every seen hotel's full gallery, but thin each held-out hotel
        # down to `sparse_k` image(s) so it is under-represented, not absent.
        seen_part = gallery[~gallery.hotel_id.isin(unseen)]
        held_part = (gallery[gallery.hotel_id.isin(unseen)]
                     .sort_values("image_id")
                     .groupby("hotel_id", group_keys=False)
                     .head(sparse_k))
        gallery = pd.concat([seen_part, held_part], ignore_index=True)
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
    ap.add_argument("--sparse-k", type=int, default=1,
                    help="gallery images kept per held-out hotel in unseen split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = evaluate(args.embeddings, args.manifest, args.split, args.sparse_k)
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()