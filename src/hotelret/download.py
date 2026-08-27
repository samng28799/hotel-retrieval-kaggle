"""Download and resize images from a manifest. Resumable.

Re-running skips files already on disk, so a dropped Colab session costs only
the unfetched remainder. Images are resized so the short side is `target`
pixels and saved as quality-92 JPEG, which keeps gallery and query images at a
comparable scale (the resolution gap is an experiment variable, handled in
`embed`, not silently here).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (research; hotel-recognition coursework)"}


def _fetch_one(row, out_root: Path, target: int):
    dest = out_root / str(row.hotel_id) / f"{row.image_id}.jpg"
    if dest.exists():
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(row.image_url, timeout=15, headers=HEADERS)
        if not r.ok:
            return f"http_{r.status_code}"
        # lazy import so the module loads without pillow for tests
        from PIL import Image
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        w, h = img.size
        scale = target / min(w, h)
        if scale < 1:
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        img.save(dest, "JPEG", quality=92)
        return "ok"
    except Exception as e:
        return type(e).__name__


def run(manifest_csv, out_dir="data/images", target=256, workers=24):
    df = pd.read_csv(manifest_csv, dtype=str)
    out_root = Path(out_dir)
    rows = list(df.itertuples())
    with cf.ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda r: _fetch_one(r, out_root, target), rows))
    counts = pd.Series(results).value_counts().to_dict()
    return counts


def main():
    ap = argparse.ArgumentParser(description="Download + resize from a manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--target", type=int, default=256, help="short-side pixels")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    counts = run(args.manifest, args.out, args.target, args.workers)
    print("download results:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
