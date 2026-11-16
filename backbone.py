"""DINOv2 ViT-B/14 feature extractor with partial freezing (paper Sec. 3.5, 3.9)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TinyViT(nn.Module):
    """Small randomly initialised ViT used only for offline smoke tests / CI."""

    def __init__(self, img_size: int = 224, patch: int = 14, dim: int = 192, depth: int = 4, heads: int = 3):
        super().__init__()
        self.embed_dim = dim
        self.patch_size = patch
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        g = max(1, img_size // patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x):
        t = self.patch_embed(x)                      # (B, D, gh, gw)
        gh, gw = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2)             # (B, N, D)
        pos = self.pos_embed
        if pos.shape[1] != gh * gw:
            g0 = int(round(pos.shape[1] ** 0.5))
            pos = pos.reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(gh, gw), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        t = t + pos
        for blk in self.blocks:
            t = blk(t)
        return {"x_norm_patchtokens": self.norm(t)}


class DinoBackbone(nn.Module):
    """Returns patch tokens (B, N, D) for images in [0, 1]."""

    def __init__(self, name: str = "dinov2_vitb14", pretrained: bool = True, frozen_blocks: int = 8,
                 img_size: int = 224, hub_dir=None):
        super().__init__()
        if name == "tiny":
            self.vit = _TinyViT(img_size=img_size)
        else:
            if hub_dir:
                self.vit = torch.hub.load(hub_dir, name, source="local", pretrained=pretrained)
            else:
                self.vit = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = 14
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.set_frozen(frozen_blocks)

    def set_frozen(self, k: int):
        """Freeze embeddings and the first k transformer blocks (k < 0 freezes everything)."""
        n_blocks = len(self.vit.blocks)
        freeze_all = k < 0
        k = n_blocks if freeze_all else min(k, n_blocks)
        embed_names = {"patch_embed", "cls_token", "pos_embed", "mask_token", "register_tokens"}
        for name, p in self.vit.named_parameters():
            top = name.split(".")[0]
            if freeze_all or top in embed_names:
                p.requires_grad = False
            elif top == "blocks":
                p.requires_grad = int(name.split(".")[1]) >= k
            else:  # final norm etc.
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.vit.forward_features(x)["x_norm_patchtokens"]
