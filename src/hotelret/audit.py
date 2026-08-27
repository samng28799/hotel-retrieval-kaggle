"""Data feasibility audit — the GO/NO-GO gate.

Two questions decide whether the project can use Hotels-50K as specified:

1. Are the image URLs still alive? (the data ships as links, not images)
2. Does the loss cluster *by hotel*? Even 25% loss is harmless if it thins
   every gallery evenly, but fatal if it deletes whole hotels.

Aggregate liveness alone is the wrong gate. This module measures per-hotel
survival and tests it against the binomial spread expected under independent
loss. Run it before downloading anything.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
from pathlib import Path

import numpy as np
import requests

from . import data

HEADERS = {"User-Agent": "Mozilla/5.0 (research; hotel-recognition coursework)"}

# go/no-go thresholds
MIN_ALIVE = 0.80        # below this, aggregate loss alone kills it
OVERDISPERSION_WARN = 5.0   # observed/binomial variance ratio above this = clustered


def _alive(url: str, timeout: int = 12) -> bool:
    """HEAD-check one URL, falling back to a ranged GET if HEAD is refused."""
    try:
        r = requests.head(url, timeout=timeout, headers=HEADERS, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, timeout=timeout, headers=HEADERS,
                             stream=True, allow_redirects=True)
            r.close()
        return bool(r.ok)
    except Exception:
        return False


def probe_sources(train, n_per_source: int = 300, workers: int = 16, seed: int = 0):
    """Aggregate liveness by image source."""
    rng = random.Random(seed)
    report = {}
    for src, grp in train.groupby("image_source"):
        urls = grp.image_url.tolist()
        sample = rng.sample(urls, min(n_per_source, len(urls)))
        with cf.ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(_alive, sample))
        report[src] = {
            "sampled": len(sample),
            "alive": int(sum(results)),
            "rate": round(sum(results) / len(sample), 4),
        }
    return report


def probe_clustering(train, gallery_hotels, n_hotels: int = 40,
                     workers: int = 24, seed: int = 0):
    """Probe *every* URL for a sample of hotels; test if loss clusters.

    Returns a dict with the overdispersion ratio: observed per-hotel survival
    variance divided by the binomial variance expected under independent loss.
    Ratio near 1 => random loss (safe). Ratio >> 1 => whole hotels vanish.
    """
    rng = random.Random(seed)
    probe = rng.sample(sorted(gallery_hotels), min(n_hotels, len(gallery_hotels)))
    sub = train[train.hotel_id.isin(probe)].copy()

    with cf.ThreadPoolExecutor(workers) as ex:
        sub["alive"] = list(ex.map(_alive, sub.image_url))

    g = sub.groupby("hotel_id").agg(n=("alive", "size"), k=("alive", "sum"))
    g["surv"] = g.k / g.n
    p_hat = g.k.sum() / g.n.sum()

    exp_var = float((p_hat * (1 - p_hat) / g.n).mean())
    obs_var = float(g.surv.var(ddof=1))
    ratio = obs_var / exp_var if exp_var > 0 else float("nan")

    return {
        "hotels_probed": int(len(g)),
        "urls_probed": int(len(sub)),
        "pooled_survival": round(float(p_hat), 4),
        "per_hotel_survival_median": round(float(g.surv.median()), 3),
        "per_hotel_survival_min": round(float(g.surv.min()), 3),
        "hotels_below_5_images": int((g.k < 5).sum()),
        "observed_variance": round(obs_var, 5),
        "binomial_variance": round(exp_var, 5),
        "overdispersion_ratio": round(ratio, 2),
    }


def verdict(sources: dict, clustering: dict) -> str:
    worst = min(s["rate"] for s in sources.values())
    dead = clustering["hotels_below_5_images"]
    total = clustering["hotels_probed"]
    ratio = clustering["overdispersion_ratio"]

    if worst < MIN_ALIVE and ratio > OVERDISPERSION_WARN:
        return ("STOP: low liveness AND clustered loss. Switch to Kaggle Hotel-ID.")
    if ratio > OVERDISPERSION_WARN or dead / total > 0.10:
        return ("PROCEED WITH OVER-SELECTION: loss clusters by hotel. "
                f"Select ~{int(1/(1-dead/total)*1.1*1000)} hotels to net 1000 usable, "
                "and drop hotels under 5 images after download.")
    return ("GO: loss is diffuse. Raise the per-hotel image cap to compensate "
            "and drop the rare sub-5-image hotel after download.")


def main():
    ap = argparse.ArgumentParser(description="Hotels-50K feasibility audit")
    ap.add_argument("--metadata", required=True, help="dir with train_set.csv etc.")
    ap.add_argument("--probe", type=int, default=300, help="URLs per source")
    ap.add_argument("--cluster-hotels", type=int, default=40)
    ap.add_argument("--min-train", type=int, default=15,
                    help="min train images for a hotel to be eligible")
    ap.add_argument("--out", default="results/audit.json")
    args = ap.parse_args()

    train = data.load_train(args.metadata)
    test = data.load_test(args.metadata)
    per = data.images_per_hotel(train)

    # candidate gallery hotels = in test set, with enough train images
    cand = [h for h in test.hotel_id.unique() if per.get(h, 0) >= args.min_train]

    print(f"eligible hotels: {len(cand):,}")
    print(f"probing {args.probe} URLs per source for liveness...")
    sources = probe_sources(train, args.probe)
    for src, s in sources.items():
        print(f"  {src:16s} {s['rate']:.1%} alive ({s['alive']}/{s['sampled']})")

    print(f"probing all URLs for {args.cluster_hotels} hotels (clustering test)...")
    clustering = probe_clustering(train, cand[:2000], args.cluster_hotels)
    print(f"  pooled survival     : {clustering['pooled_survival']:.1%}")
    print(f"  overdispersion ratio: {clustering['overdispersion_ratio']}x binomial")
    print(f"  hotels < 5 images   : {clustering['hotels_below_5_images']}"
          f"/{clustering['hotels_probed']}")

    v = verdict(sources, clustering)
    print("\n=== VERDICT ===")
    print(v)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"sources": sources, "clustering": clustering, "verdict": v}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
