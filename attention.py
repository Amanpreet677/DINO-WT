"""Frequency-guided attention (paper Sec. 3.6, Eq. 16-21, Algorithm 1).

Queries come from the semantic (DINO) tokens. A frequency embedding E(Wf) built
from the wavelet prior is added to the key/value inputs, and the attention
logits receive an additive per-head bias  lambda_h * Wf_key  so that keys located
in high-frequency (in-focus) regions get more attention. The output is merged
back through a residual connection  F_out = F_s + lambda * F_f  (Eq. 21).

The two images' tokens are concatenated into one sequence (the "joint
representation"), so every token can attend to both source images.
"""
import torch
import torch.nn as nn


class FrequencyGuidedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, lambda_init: float = 1.0, mlp_ratio: float = 2.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim, self.h, self.dh = dim, num_heads, dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.freq_embed = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.lam = nn.Parameter(torch.full((num_heads,), float(lambda_init)))  # per-head guidance strength
        self.res_scale = nn.Parameter(torch.tensor(1.0))                        # lambda in Eq. 21
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, D) semantic tokens; prior: (B, L, 1) frequency prior in [0, 1]."""
        b, l, d = tokens.shape
        x = self.norm1(tokens)
        fe = self.freq_embed(prior)
        q, k, v = self.q(x), self.k(x + fe), self.v(x + fe)

        def split(t):
            return t.reshape(b, l, self.h, self.dh).transpose(1, 2)  # (B, h, L, dh)

        q, k, v = split(q), split(k), split(v)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.dh ** -0.5)       # (B, h, L, L)
        bias = self.lam.view(1, self.h, 1, 1) * prior.reshape(b, 1, 1, l)          # (B, h, 1, L)
        attn = torch.softmax(logits + bias, dim=-1)
        ff = torch.matmul(attn, v).transpose(1, 2).reshape(b, l, d)
        ff = self.proj(ff)
        out = tokens + self.res_scale * ff
        return out + self.mlp(self.norm2(out))
