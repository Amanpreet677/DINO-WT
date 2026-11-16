"""DINO-WT: Frequency-guided self-supervised ViT with learnable wavelet representation for MFIF."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FrequencyGuidedAttention
from .backbone import DinoBackbone
from .decoder import LightweightDecoder
from .wavelet import LearnableWavelet


class DINOWT(nn.Module):
    def __init__(self, backbone: DinoBackbone, num_heads=None, lambda_init=1.0,
                 decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        dim = backbone.embed_dim
        heads = num_heads or max(1, dim // 64)
        self.backbone = backbone
        self.patch = backbone.patch_size
        self.wavelet = LearnableWavelet(3)
        self.attention = FrequencyGuidedAttention(dim, heads, lambda_init)
        self.decoder = LightweightDecoder(dim, tuple(decoder_channels))
        self.gamma_logit = nn.Parameter(torch.zeros(1))  # mixes pixel-domain and wavelet-domain fusion

    def forward(self, ia: torch.Tensor, ib: torch.Tensor) -> dict:
        """ia, ib: (B, 3, H, W) in [0, 1]; H and W must be multiples of 14 (and even)."""
        b, _, h, w = ia.shape
        gh, gw = h // self.patch, w // self.patch

        # Phase 2 - learnable wavelet decomposition + frequency prior
        wa, wb = self.wavelet(ia), self.wavelet(ib)                      # (B,3,4,h/2,w/2)
        pa, pb = self.wavelet.frequency_prior(wa), self.wavelet.frequency_prior(wb)  # (B,1,h/2,w/2)

        # Phase 3 - DINO semantic tokens
        fa, fb = self.backbone(ia), self.backbone(ib)                    # (B,N,D)
        n = fa.shape[1]

        # Token-grid priors
        pga = F.adaptive_avg_pool2d(pa, (gh, gw))
        pgb = F.adaptive_avg_pool2d(pb, (gh, gw))
        prior = torch.cat([pga.flatten(2).transpose(1, 2), pgb.flatten(2).transpose(1, 2)], dim=1)  # (B,2N,1)

        # Phase 4 - frequency-guided attention over the joint token sequence
        out = self.attention(torch.cat([fa, fb], dim=1), prior)
        oa, ob = out[:, :n], out[:, n:]

        # Phase 5 - decoder -> decision map -> fusion
        pfa = F.interpolate(pa, size=(h, w), mode="bilinear", align_corners=False)
        pfb = F.interpolate(pb, size=(h, w), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(self.decoder(oa, ob, pga, pgb, ia, ib, pfa, pfb, (gh, gw)))   # (B,1,H,W)

        i_pix = mask * ia + (1.0 - mask) * ib                                   # Eq. 45
        m_half = F.avg_pool2d(mask, 2).unsqueeze(2)                              # (B,1,1,h/2,w/2)
        i_wav = self.wavelet.inverse(m_half * wa + (1.0 - m_half) * wb)          # sub-band fusion + Psi^-1
        g = torch.sigmoid(self.gamma_logit)
        fused = (1.0 - g) * i_pix + g * i_wav

        return {"fused": fused, "mask": mask, "wavelet_pr": self.wavelet.pr_loss(ia)}

    @torch.no_grad()
    def fuse(self, ia: torch.Tensor, ib: torch.Tensor):
        """Inference on arbitrary image sizes: resize to a multiple of 14, fuse, resize back."""
        h, w = ia.shape[-2:]
        m = self.patch
        nh, nw = max(m, round(h / m) * m), max(m, round(w / m) * m)
        if (nh, nw) != (h, w):
            ia = F.interpolate(ia, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
            ib = F.interpolate(ib, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        out = self.forward(ia, ib)
        fused, mask = out["fused"].clamp(0, 1), out["mask"]
        if (nh, nw) != (h, w):
            fused = F.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        return fused, mask


def build_model(cfg: dict) -> DINOWT:
    backbone = DinoBackbone(
        name=cfg.get("backbone", "dinov2_vitb14"),
        pretrained=cfg.get("pretrained", True),
        frozen_blocks=cfg.get("frozen_blocks", 8),
        img_size=cfg.get("img_size", 224),
        hub_dir=cfg.get("hub_dir"),
    )
    return DINOWT(backbone, num_heads=cfg.get("num_heads"),
                  lambda_init=cfg.get("attn_lambda_init", 1.0),
                  decoder_channels=cfg.get("decoder_channels", (256, 128, 64, 32)))
