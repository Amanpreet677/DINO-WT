"""Datasets.

* SyntheticMFIFDataset - builds multi-focus pairs on the fly from any folder of clean images
  (COCO, PASCAL VOC 2012, ...) using spatially varying Gaussian defocus (Eq. 36). Ground truth = clean image.
* RealMFIFDataset      - loads registered pairs from Lytro / MFFW style folders (no ground truth).
"""
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .preprocess import histogram_equalize, preprocess_pair

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: str):
    files = [p for p in sorted(Path(root).rglob("*")) if p.suffix.lower() in IMG_EXT]
    if not files:
        raise FileNotFoundError(f"No images found under '{root}'")
    return files


def random_focus_mask(size: int, rng: np.random.Generator) -> torch.Tensor:
    """Smooth random binary-ish region mask (1,H,W): 1 = region in focus in image A."""
    cells = int(rng.integers(3, 7))
    noise = torch.from_numpy(rng.random((1, 1, cells, cells)).astype(np.float32))
    noise = F.interpolate(noise, size=(size, size), mode="bicubic", align_corners=False)
    q = torch.quantile(noise.flatten(), float(rng.uniform(0.35, 0.65)))
    mask = (noise > q).float()
    mask = TF.gaussian_blur(mask, [9, 9], [2.0, 2.0])
    return mask[0]


class SyntheticMFIFDataset(Dataset):
    def __init__(self, root, size=224, train=True, sigma_range=(2.0, 5.0), seed=0,
                 max_images=None, hist_eq=False):
        self.files = list_images(root)
        if max_images:
            self.files = self.files[:max_images]
        self.size, self.train, self.sigma_range, self.seed, self.hist_eq = size, train, sigma_range, seed, hist_eq
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2),
            ])
        else:
            self.tf = T.Compose([T.Resize(size), T.CenterCrop(size)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        img = self.tf(img)
        if self.hist_eq:
            img = Image.fromarray(histogram_equalize(np.array(img)))
        x = TF.to_tensor(img)                                           # (3,H,W) in [0,1]
        rng = np.random.default_rng(None if self.train else self.seed + i)
        mask = random_focus_mask(self.size, rng)
        sigma = float(rng.uniform(*self.sigma_range))
        k = 2 * math.ceil(3 * sigma) + 1
        blurred = TF.gaussian_blur(x, [k, k], [sigma, sigma])
        ia = mask * x + (1 - mask) * blurred                            # focus on region
        ib = (1 - mask) * x + mask * blurred                            # focus on complement
        return ia, ib, x


def _find_pairs(root: str):
    root = Path(root)
    pairs = []
    # Layout 1: sub-folders A/B (or near/far, source1/source2) with matching file stems
    names = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    for a_name, b_name in [("a", "b"), ("near", "far"), ("source1", "source2"), ("1", "2")]:
        if a_name in names and b_name in names:
            b_files = {p.stem: p for p in list_images(names[b_name])}
            for p in list_images(names[a_name]):
                if p.stem in b_files:
                    pairs.append((p.stem, p, b_files[p.stem]))
            if pairs:
                return pairs
    # Layout 2: flat folder, files like  lytro-01-A.jpg / lytro-01-B.jpg  (also _A/_B, "A"/"B" suffix)
    groups = {}
    for p in list_images(root):
        m = re.match(r"^(.*?)[-_ ]?([abAB])$", p.stem)
        if m:
            groups.setdefault(m.group(1), {})[m.group(2).lower()] = p
    for k in sorted(groups):
        if "a" in groups[k] and "b" in groups[k]:
            pairs.append((k, groups[k]["a"], groups[k]["b"]))
    if not pairs:
        raise FileNotFoundError(
            f"Could not find image pairs in '{root}'. Use sub-folders A/ and B/ or files named *-A.* / *-B.*")
    return pairs


class RealMFIFDataset(Dataset):
    """Returns (img_A, img_B, name). If `size` is None the native resolution is kept."""

    def __init__(self, root, size=None, register=False, hist_eq=False):
        self.pairs = _find_pairs(root)
        self.size, self.register, self.hist_eq = size, register, hist_eq

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        name, pa, pb = self.pairs[i]
        a = Image.open(pa).convert("RGB")
        b = Image.open(pb).convert("RGB")
        if b.size != a.size:
            b = b.resize(a.size, Image.BICUBIC)
        if self.register or self.hist_eq:
            na, nb = preprocess_pair(np.array(a), np.array(b), self.register, self.hist_eq)
            a, b = Image.fromarray(na), Image.fromarray(nb)
        if self.size:
            a, b = a.resize((self.size, self.size), Image.BICUBIC), b.resize((self.size, self.size), Image.BICUBIC)
        return TF.to_tensor(a), TF.to_tensor(b), name
