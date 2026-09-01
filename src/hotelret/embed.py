"""Turn images into L2-normalized embedding vectors, cached to .npy.

Encoders are pluggable via a small registry so adding a model is one function.
The heavy deps (torch, timm, open_clip) are imported lazily inside each loader
so the module imports cleanly in a bare environment and in tests.

Cache format: a single float32 array `embeddings.npy` (N x D) plus a sidecar
`embeddings.ids.json` giving the image_id order, so vectors always line up with
the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---- encoder registry -----------------------------------------------------

_ENCODERS = {}


def register(name):
    def deco(fn):
        _ENCODERS[name] = fn
        return fn
    return deco


def available():
    return sorted(_ENCODERS)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@register("dinov2")
def _load_dinov2():
    import torch
    dev = _device()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval().to(dev)
    dim = 1024

    def encode(batch):  # batch: (B,3,H,W) tensor already normalized
        with torch.no_grad():
            return model(batch.to(dev)).cpu().numpy()
    return encode, dim


@register("clip")
def _load_clip():
    import torch
    import open_clip
    dev = _device()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    model.eval().to(dev)
    dim = 768

    def encode(batch):
        with torch.no_grad():
            return model.encode_image(batch.to(dev)).cpu().numpy()
    return encode, dim


@register("sscd")
def _load_sscd():
    """SSCD released as a standalone TorchScript model — no SSCD code needed."""
    import torch
    dev = _device()
    weights = Path("external/sscd_disc_mixup.torchscript.pt")
    if not weights.exists():
        raise FileNotFoundError(
            "Download sscd_disc_mixup.torchscript.pt from "
            "github.com/facebookresearch/sscd-copy-detection into external/")
    model = torch.jit.load(str(weights))
    model.eval().to(dev)
    dim = 512

    def encode(batch):
        with torch.no_grad():
            return model(batch.to(dev)).cpu().numpy()
    return encode, dim


# ---- driver ---------------------------------------------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def extract(image_dir, model_name, out_path, batch_size=64, target=224,
            only_ids=None):
    """Extract embeddings for images under image_dir.

    If ``only_ids`` is given (an iterable of image_id stems), only those images
    are embedded — use it to skip the ~half of the Kaggle set that isn't in the
    manifest. Lazy torch import so the module loads in bare/test environments.
    """
    import torch
    from PIL import Image
    import torchvision.transforms as T

    if model_name not in _ENCODERS:
        raise ValueError(f"unknown model {model_name!r}; have {available()}")

    encode, dim = _ENCODERS[model_name]()

    tf = T.Compose([
        T.Resize(target),
        T.CenterCrop(target),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    paths = sorted(Path(image_dir).rglob("*.jpg"))
    if only_ids is not None:
        wanted = set(only_ids)
        paths = [p for p in paths if p.stem in wanted]
    ids = [p.stem for p in paths]

    vecs = []
    batch = []
    for p in paths:
        batch.append(tf(Image.open(p).convert("RGB")))
        if len(batch) == batch_size:
            vecs.append(encode(torch.stack(batch)))
            batch = []
    if batch:
        vecs.append(encode(torch.stack(batch)))

    emb = l2_normalize(np.concatenate(vecs).astype("float32"))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    out.with_suffix(".ids.json").write_text(json.dumps(ids))
    return emb.shape


def main():
    ap = argparse.ArgumentParser(description="Extract image embeddings")
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True, choices=available())
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    shape = extract(args.images, args.model, args.out, args.batch_size)
    print(f"wrote {args.out}: {shape[0]} vectors x {shape[1]} dims")


if __name__ == "__main__":
    main()