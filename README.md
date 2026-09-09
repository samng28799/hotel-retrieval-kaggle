# Hotel Recognition for Unseen Properties

Comparing image-retrieval models on Kaggle (https://www.kaggle.com/c/hotel-id-to-combat-human-trafficking-2022-fgvc9), with a focus on how accuracy
holds up on hotels the model never saw during training.


## What this does

Given a photo taken inside a hotel, rank a gallery of known hotels by visual
similarity and return the most likely property. The project compares several
image encoders (DINOv2, CLIP, SSCD, and a trained ArcFace CNN) under one
**unseen-hotel** evaluation split, and measures two effects the standard
benchmark ignores: the resolution gap between gallery and query photos, and the
decay of the dataset over time (it ships as URLs, not images).

## Pipeline

```
audit → manifest → download → embed → index → evaluate
```

Each stage writes to disk so the expensive steps (download, embed) run once and
everything after is cheap.

| Stage | Module | Output |
|-------|--------|--------|
| 1. Audit link liveness | `hotelret.audit` | `results/audit.json` |
| 2. Build subset manifest | `hotelret.manifest` | `data/manifest/*.csv` |
| 3. Download + resize | `hotelret.download` | `data/images/…` |
| 4. Extract embeddings | `hotelret.embed` | `data/embeddings/*.npy` |
| 5. Build FAISS index | `hotelret.index` | in-memory / `.faiss` |
| 6. Evaluate retrieval | `hotelret.evaluate` | `results/metrics_*.json` |

## Quickstart

```bash
# 1. install
pip install -e .

# 2. get the dataset metadata (14 MB, ships in the official repo)
https://www.kaggle.com/c/hotel-id-to-combat-human-trafficking-2022-fgvc9

# 3. GO/NO-GO: is the data still reachable, and does loss cluster by hotel?
python -m hotelret.audit --metadata external/Hotels-50K/input/dataset --probe 300

# 4. build the working subset (1000 hotels)
python -m hotelret.manifest --metadata external/Hotels-50K/input/dataset --hotels 1000

# 5. download (resumable) — only after audit says GO
python -m hotelret.download --manifest data/manifest/gallery.csv --out data/images

# 6. embed with a chosen encoder
python -m hotelret.embed --images data/images --model dinov2 --out data/embeddings/dinov2.npy

# 7. evaluate on the unseen-hotel split
python -m hotelret.evaluate --embeddings data/embeddings/dinov2.npy --split unseen
```

## Notebooks

- `notebooks/00_feasibility.ipynb` — the data audit as an interactive report
  (run this first, on Colab, before committing to the download).

## Data source

**Primary: Kaggle Hotel-ID 2022 (FGVC9).** The Hotels-50K feasibility audit
returned STOP — 67% link survival with 15x overdispersion, meaning whole hotels
have vanished from the source CDN rather than each gallery thinning evenly. We
therefore use the Kaggle dataset, which ships decoded image files and cannot
decay. The measured decay of Hotels-50K is itself reported as a finding.

`hotelret.kaggle_data` handles both the competition layout and the pre-resized
256x256 mirror. The audit code (`hotelret.audit`) is kept for reproducibility.

## Status

- [x] Repo scaffold
- [x] Round-2 audit run — **Hotels-50K STOP** (clustered link loss, 15x overdispersion)
- [x] Kaggle adapter + tests (both dataset layouts)
- [x] Colab run-all notebook (Kaggle path)
- [ ] Download Kaggle subset
- [ ] Baseline embeddings (DINOv2, CLIP, SSCD)
- [ ] Trained ArcFace baseline
- [ ] Unseen-hotel evaluation
- [ ] Report

## License

Code: MIT. The Hotels-50K data and the third-party models keep their own
licenses (see `NOTICE`).
