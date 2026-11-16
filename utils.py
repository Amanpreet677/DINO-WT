import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply 'a.b.c=value' overrides (value parsed as YAML)."""
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.n += n

    @property
    def avg(self):
        return self.sum / max(1, self.n)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_model_from_checkpoint(ckpt_path: str, device, override_model_cfg=None):
    """Rebuild a DINOWT model from a checkpoint saved by train.py."""
    from .models import build_model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = dict(ckpt["config"]["model"])
    mcfg["pretrained"] = False          # weights come from the checkpoint
    if override_model_cfg:
        mcfg.update(override_model_cfg)
    model = build_model(mcfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
