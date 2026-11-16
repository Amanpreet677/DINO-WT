import pytest

torch = pytest.importorskip("torch")

from dinowt.losses import DinoWTLoss
from dinowt.models import LearnableWavelet, build_model


def _cfg():
    return {"backbone": "tiny", "pretrained": False, "frozen_blocks": 2, "img_size": 56,
            "decoder_channels": [32, 16, 16, 8]}


def test_wavelet_perfect_reconstruction():
    wt = LearnableWavelet(3)
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(wt.inverse(wt(x)), x, atol=1e-5)
    prior = wt.frequency_prior(wt(x))
    assert prior.shape == (2, 1, 16, 16) and prior.max() <= 1.0 + 1e-6


def test_forward_backward_and_fuse():
    model = build_model(_cfg())
    ia, ib, gt = torch.rand(2, 3, 56, 56), torch.rand(2, 3, 56, 56), torch.rand(2, 3, 56, 56)
    out = model(ia, ib)
    assert out["fused"].shape == ia.shape and out["mask"].shape == (2, 1, 56, 56)
    enc = build_model(_cfg()).backbone
    enc.set_frozen(-1)
    loss_fn = DinoWTLoss(model.wavelet, enc)
    loss, terms = loss_fn(out["fused"], gt, out)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.attention.parameters())
    fused, mask = model.eval().fuse(torch.rand(1, 3, 100, 130), torch.rand(1, 3, 100, 130))
    assert fused.shape == (1, 3, 100, 130)
