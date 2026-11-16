"""Evaluate a trained DINO-WT checkpoint with the ten metrics of the paper.

Real pairs (Lytro / MFFW; no ground truth; reference metrics averaged over both sources):
    python evaluate.py --ckpt runs/dinowt/best.pt --data real --root data/Lytro --out results/lytro

Synthetic pairs (COCO / VOC; ground truth = original image):
    python evaluate.py --ckpt runs/dinowt/best.pt --data synthetic --root data/val --out results/coco
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dinowt.data import RealMFIFDataset, SyntheticMFIFDataset
from dinowt.metrics import METRIC_NAMES, evaluate_fusion
from dinowt.utils import ensure_dir, get_device, load_model_from_checkpoint


def to_np(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def save_img(arr, path):
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["real", "synthetic"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--size", type=int, default=None, help="resize inputs to SxS (default: native for real, 224 synthetic)")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--hub_dir", default=None, help="local dinov2 repo for offline loading")
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_model_from_checkpoint(args.ckpt, device, {"hub_dir": args.hub_dir} if args.hub_dir else None)
    out_dir = ensure_dir(args.out)

    if args.data == "real":
        ds = RealMFIFDataset(args.root, size=args.size)
    else:
        cfg = ckpt["config"]
        ds = SyntheticMFIFDataset(args.root, args.size or cfg["model"]["img_size"], train=False,
                                  sigma_range=tuple(cfg["data"]["sigma_range"]), seed=123,
                                  max_images=args.max_images)
    n = len(ds) if not args.max_images else min(len(ds), args.max_images)

    rows = []
    for i in tqdm(range(n), desc="evaluating"):
        if args.data == "real":
            ia, ib, name = ds[i]
            gt = None
        else:
            ia, ib, gt = ds[i]
            name = f"{i:05d}"
        fused, _ = model.fuse(ia.unsqueeze(0).to(device), ib.unsqueeze(0).to(device))
        f_np, a_np, b_np = to_np(fused), to_np(ia.unsqueeze(0)), to_np(ib.unsqueeze(0))
        m = evaluate_fusion(f_np, a_np, b_np, None if gt is None else to_np(gt.unsqueeze(0)))
        rows.append({"name": name, **m})
        if args.save_images:
            save_img(f_np, os.path.join(out_dir, f"{name}_fused.png"))

    with open(os.path.join(out_dir, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + METRIC_NAMES)
        w.writeheader()
        w.writerows(rows)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_NAMES}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nMean over", len(rows), "image pairs")
    for k in METRIC_NAMES:
        print(f"  {k:5s}: {summary[k]:.4f}")


if __name__ == "__main__":
    main()
