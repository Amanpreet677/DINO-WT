"""Fuse a pair of multi-focus images (any size) into one all-in-focus image.

    python infer.py --ckpt runs/dinowt/best.pt --a near.jpg --b far.jpg --out fused.png
Folder of pairs (A/B sub-folders or *-A.jpg / *-B.jpg naming):
    python infer.py --ckpt runs/dinowt/best.pt --folder data/Lytro --out results/lytro_fused
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from dinowt.data import RealMFIFDataset
from dinowt.utils import ensure_dir, get_device, load_model_from_checkpoint


def save(t, path):
    arr = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--folder")
    ap.add_argument("--out", default="fused.png")
    ap.add_argument("--save_mask", action="store_true")
    ap.add_argument("--hub_dir", default=None)
    args = ap.parse_args()

    device = get_device()
    model, _ = load_model_from_checkpoint(args.ckpt, device, {"hub_dir": args.hub_dir} if args.hub_dir else None)

    jobs = []
    if args.folder:
        ds = RealMFIFDataset(args.folder)
        ensure_dir(args.out)
        for i in range(len(ds)):
            ia, ib, name = ds[i]
            jobs.append((ia, ib, os.path.join(args.out, f"{name}_fused.png")))
    else:
        if not (args.a and args.b):
            ap.error("give --a and --b, or --folder")
        a = Image.open(args.a).convert("RGB")
        b = Image.open(args.b).convert("RGB").resize(a.size, Image.BICUBIC)
        jobs.append((TF.to_tensor(a), TF.to_tensor(b), args.out))

    for ia, ib, path in jobs:
        fused, mask = model.fuse(ia.unsqueeze(0).to(device), ib.unsqueeze(0).to(device))
        save(fused, path)
        if args.save_mask:
            save(mask.repeat(1, 3, 1, 1), path.replace(".png", "_mask.png"))
        print("saved", path)


if __name__ == "__main__":
    main()
