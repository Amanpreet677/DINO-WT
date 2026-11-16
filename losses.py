"""Composite loss (paper Sec. 3.8, Eq. 27-32):
L = a*L_pixel + b*L_SSIM + g*L_freq + d*L_perc + e*L_edge  (+ wavelet perfect-reconstruction term)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = torch.outer(g, g)
    return w2d.expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM between two (B, C, H, W) batches."""
    x, y = x.float(), y.float()
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    sy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))
    return m.mean()


class Sobel(nn.Module):
    """Image gradient operator (nabla in Eq. 31): Sobel x/y per channel."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        self.register_buffer("k", torch.stack([kx, kx.t()]).unsqueeze(1))  # (2,1,3,3)

    def forward(self, x):
        b, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * c, 1, h, w), self.k.to(x.dtype), padding=1)
        return y.view(b, c * 2, h, w)


class DinoWTLoss(nn.Module):
    def __init__(self, wavelet: nn.Module, encoder=None, weights: dict = None):
        super().__init__()
        w = {"pixel": 1.0, "ssim": 1.0, "freq": 0.5, "perceptual": 0.1, "edge": 0.5, "wavelet_pr": 1.0}
        w.update(weights or {})
        self.w = w
        self.wavelet = wavelet            # shared with the model (learnable Psi)
        self.encoder = encoder            # frozen DINO encoder for the perceptual loss
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.sobel = Sobel()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def forward(self, fused, ref, aux=None):
        fused, ref = fused.float(), ref.float()
        terms = {
            "pixel": F.l1_loss(fused, ref),                                   # Eq. 27
            "ssim": 1.0 - ssim(fused, ref),                                   # Eq. 28
            "freq": F.l1_loss(self.wavelet(fused), self.wavelet(ref)),        # Eq. 29
            "edge": F.l1_loss(self.sobel(fused), self.sobel(ref)),            # Eq. 31
        }
        if self.encoder is not None and self.w["perceptual"] > 0:             # Eq. 30
            with torch.no_grad():
                f_ref = self.encoder(ref)
            terms["perceptual"] = F.mse_loss(self.encoder(fused).float(), f_ref.float())
        if aux is not None and "wavelet_pr" in aux:
            terms["wavelet_pr"] = aux["wavelet_pr"].float()
        total = sum(self.w[k] * v for k, v in terms.items())                  # Eq. 32
        terms["total"] = total
        return total, terms
