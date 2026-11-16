"""Learnable wavelet transform (paper Sec. 3.4 and 3.7).

A single-level 2x2 stride-2 convolutional filter bank, initialised to the Haar
wavelet and trained end-to-end. The analysis bank produces LL, LH, HL, HH
sub-bands per colour channel; the synthesis bank (transposed conv) is the
learnable inverse transform Psi^-1.
"""
import torch
import torch.nn as nn

# Orthonormal Haar filters: LL, LH, HL, HH
_HAAR = torch.tensor([
    [[1.0, 1.0], [1.0, 1.0]],      # LL  low-pass
    [[1.0, 1.0], [-1.0, -1.0]],    # LH  horizontal edges
    [[1.0, -1.0], [1.0, -1.0]],    # HL  vertical edges
    [[1.0, -1.0], [-1.0, 1.0]],    # HH  diagonal edges
]) / 2.0


class LearnableWavelet(nn.Module):
    def __init__(self, in_ch: int = 3, learnable: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.analysis = nn.Conv2d(in_ch, 4 * in_ch, 2, stride=2, groups=in_ch, bias=False)
        self.synthesis = nn.ConvTranspose2d(4 * in_ch, in_ch, 2, stride=2, groups=in_ch, bias=False)
        with torch.no_grad():
            w = _HAAR.unsqueeze(1).repeat(in_ch, 1, 1, 1)  # (4*in_ch, 1, 2, 2)
            self.analysis.weight.copy_(w)
            self.synthesis.weight.copy_(w)
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) with even H, W  ->  coeffs (B, C, 4, H/2, W/2) ordered LL, LH, HL, HH."""
        y = self.analysis(x)
        b, _, h, w = y.shape
        return y.view(b, self.in_ch, 4, h, w)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """coeffs (B, C, 4, h, w) -> image (B, C, 2h, 2w)."""
        b, c, _, h, w = coeffs.shape
        return self.synthesis(coeffs.reshape(b, c * 4, h, w))

    @staticmethod
    def frequency_prior(coeffs: torch.Tensor) -> torch.Tensor:
        """Eq. 10-11: Wf = (|LH|+|HL|+|HH|) / max(.), averaged over colour channels.
        Returns (B, 1, h, w) with values in [0, 1]."""
        hf = coeffs[:, :, 1:].abs().sum(dim=2)            # (B, C, h, w)
        prior = hf.mean(dim=1, keepdim=True)              # (B, 1, h, w)
        denom = prior.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return prior / denom

    def pr_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Perfect-reconstruction regulariser: ||Psi^-1(Psi(x)) - x||_1."""
        return torch.mean(torch.abs(self.inverse(self(x)) - x))
