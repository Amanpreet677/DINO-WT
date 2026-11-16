"""Lightweight convolutional decoder / fusion head (paper Sec. 3.7)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(cin, cout):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class LightweightDecoder(nn.Module):
    """Aggregates fused tokens + frequency priors (Eq. 22-24) and predicts a focus decision map."""

    def __init__(self, dim: int, channels=(256, 128, 64, 32)):
        super().__init__()
        c0 = channels[0]
        self.aggregate = nn.Sequential(                       # 1x1 conv + BN + GELU  (phi, Eq. 23)
            nn.Conv2d(2 * dim + 2, c0, 1, bias=False), nn.BatchNorm2d(c0), nn.GELU())
        ups, c = [], c0
        for nc in channels[1:]:
            ups.append(_up(c, nc))
            c = nc
        self.ups = nn.Sequential(*ups)
        self.refine = nn.Sequential(                          # full-resolution refinement
            nn.Conv2d(c + 2 + 6, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 3, padding=1))

    @staticmethod
    def _grid(t, gh, gw):
        b, n, d = t.shape
        return t.transpose(1, 2).reshape(b, d, gh, gw)

    def forward(self, oa, ob, pga, pgb, ia, ib, pfa, pfb, grid):
        """oa, ob: (B,N,D) fused tokens; pga, pgb: (B,1,gh,gw) token-grid priors;
        ia, ib: (B,3,H,W) images; pfa, pfb: (B,1,H,W) full-res priors; grid=(gh,gw)."""
        gh, gw = grid
        x = torch.cat([self._grid(oa, gh, gw), self._grid(ob, gh, gw), pga, pgb], dim=1)
        x = self.ups(self.aggregate(x))
        x = F.interpolate(x, size=ia.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, pfa, pfb, ia, ib], dim=1)
        return self.refine(x)                                  # decision-map logits (B,1,H,W)
