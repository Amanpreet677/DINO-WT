import numpy as np
from dinowt.metrics import METRIC_NAMES, evaluate_fusion, psnr, ssim_np, vifp, qabf, to_gray255


def _img(seed=0, n=96):
    r = np.random.default_rng(seed)
    base = r.random((n // 8, n // 8, 3))
    return np.kron(base, np.ones((8, 8, 1)))


def test_identity_metrics():
    x = _img()
    g = to_gray255(x)
    assert abs(ssim_np(g, g) - 1.0) < 1e-6
    assert psnr(g, g) > 90
    assert abs(vifp(g, g) - 1.0) < 1e-3


def test_qabf_range_and_self():
    x = to_gray255(_img(1))
    q = qabf(x, x, x)
    assert 0.9 < q <= 1.0


def test_evaluate_fusion_keys_and_gt():
    a, b = _img(2), _img(3)
    f = 0.5 * (a + b)
    m = evaluate_fusion(f, a, b)
    assert set(m) == set(METRIC_NAMES) and all(np.isfinite(v) for v in m.values())
    m_gt = evaluate_fusion(a, a, b, ref=a)
    assert m_gt["RMSE"] < 1e-6
