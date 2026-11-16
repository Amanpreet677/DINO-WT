"""Ten fusion metrics (paper Eq. 48): Qabf, SSIM, VIF, EN, FMI, AG, SF, MI, PSNR, RMSE.

NumPy/SciPy only. Inputs are HxWx3 float arrays in [0, 1]; metrics are computed on the luminance
channel scaled to [0, 255]. Reference-based metrics (SSIM, VIF, PSNR, RMSE) use the ground truth when
it is given, otherwise they are averaged over the two source images (standard practice when no
all-in-focus ground truth exists, e.g. Lytro / MFFW).

Notes: Qabf follows Xydeas & Petrovic; VIF is the pixel-domain multi-scale VIFp; FMI is a simplified
histogram-based variant on gradient maps (not the sliding-window Haghighat implementation).
"""
import numpy as np
from scipy import ndimage, signal


def to_gray255(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img * 255.0


def _u8(g):
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- no-reference metrics
def entropy(f):
    h = np.bincount(_u8(f).ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def avg_gradient(f):
    dx = np.diff(f, axis=1)[:-1, :]
    dy = np.diff(f, axis=0)[:, :-1]
    return float(np.mean(np.sqrt((dx ** 2 + dy ** 2) / 2.0)))


def spatial_frequency(f):
    rf = np.sqrt(np.mean(np.diff(f, axis=0) ** 2))
    cf = np.sqrt(np.mean(np.diff(f, axis=1) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def _mi(x, y, bins=256, rng=(0, 256)):
    joint, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins, range=[rng, rng])
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def mutual_information(f, a, b):
    return _mi(_u8(a), _u8(f)) + _mi(_u8(b), _u8(f))


def _grad_mag(g):
    return np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0))


def _ent_from_hist(x, bins, rng):
    h, _ = np.histogram(x.ravel(), bins=bins, range=rng)
    p = h / max(1, h.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def feature_mutual_information(f, a, b, bins=64):
    """Simplified FMI: normalised MI between Sobel gradient-magnitude maps."""
    gf, ga, gb = _grad_mag(f), _grad_mag(a), _grad_mag(b)
    m = max(gf.max(), ga.max(), gb.max(), 1e-8)
    gf, ga, gb = [np.clip(g / m * (bins - 1), 0, bins - 1) for g in (gf, ga, gb)]
    rng = (0, bins)
    out = 0.0
    for g in (ga, gb):
        denom = _ent_from_hist(gf, bins, rng) + _ent_from_hist(g, bins, rng)
        out += _mi(g, gf, bins, rng) / max(denom, 1e-8)
    return out / 2.0


def qabf(a, b, f):
    """Edge-information-preservation metric Q^{AB/F} (Xydeas & Petrovic)."""
    Lg, kg, sg = 0.9994, -15.0, 0.5
    La, ka, sa = 0.9879, -22.0, 0.8

    def edge(g):
        gx, gy = ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)
        mag = np.hypot(gx, gy)
        gx = np.where(gx == 0, 1e-12, gx)
        ang = np.arctan(gy / gx)
        return mag, ang

    def q_map(ga, aa, gf, af):
        mx, mn = np.maximum(ga, gf), np.minimum(ga, gf)
        G = np.where(mx > 0, mn / np.maximum(mx, 1e-12), 0.0)
        A = np.abs(np.abs(aa - af) - np.pi / 2) * 2 / np.pi   # 1 = identical orientation
        qg = Lg / (1 + np.exp(kg * (G - sg)))
        qa = La / (1 + np.exp(ka * (A - sa)))
        return qg * qa

    ga, aa = edge(a)
    gb, ab = edge(b)
    gf, af = edge(f)
    qaf, qbf = q_map(ga, aa, gf, af), q_map(gb, ab, gf, af)
    num = np.sum(qaf * ga + qbf * gb)
    den = np.sum(ga + gb)
    return float(num / max(den, 1e-12))


# ---------------------------------------------------------------- reference-based metrics
def ssim_np(x, y, sigma=1.5, data_range=255.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    flt = lambda z: ndimage.gaussian_filter(z, sigma, truncate=3.5)
    mx, my = flt(x), flt(y)
    sx, sy = flt(x * x) - mx * mx, flt(y * y) - my * my
    sxy = flt(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))
    return float(m.mean())


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def _gauss2d(n, sigma):
    ax = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def vifp(ref, dist, sigma_nsq=2.0):
    """Pixel-domain multi-scale VIF (Sheikh & Bovik)."""
    num = den = 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        win = _gauss2d(n, n / 5.0)
        if scale > 1:
            ref = signal.convolve2d(ref, win, mode="valid")[::2, ::2]
            dist = signal.convolve2d(dist, win, mode="valid")[::2, ::2]
        cv = lambda z: signal.convolve2d(z, win, mode="valid")
        mu1, mu2 = cv(ref), cv(dist)
        s1 = np.maximum(cv(ref * ref) - mu1 * mu1, 0)
        s2 = np.maximum(cv(dist * dist) - mu2 * mu2, 0)
        s12 = cv(ref * dist) - mu1 * mu2
        g = s12 / (s1 + 1e-10)
        sv = s2 - g * s12
        m = s1 < 1e-10
        g[m] = 0; sv[m] = s2[m]; s1[m] = 0
        m = s2 < 1e-10
        g[m] = 0; sv[m] = 0
        m = g < 0
        sv[m] = s2[m]; g[m] = 0
        sv[sv <= 1e-10] = 1e-10
        num += np.sum(np.log10(1 + g ** 2 * s1 / (sv + sigma_nsq)))
        den += np.sum(np.log10(1 + s1 / sigma_nsq))
    return float(num / max(den, 1e-12))


# ---------------------------------------------------------------- top-level
METRIC_NAMES = ["Qabf", "SSIM", "VIF", "EN", "FMI", "AG", "SF", "MI", "PSNR", "RMSE"]


def evaluate_fusion(fused, src_a, src_b, ref=None) -> dict:
    """All inputs HxWx3 float in [0,1]. `ref` = optional all-in-focus ground truth."""
    f, a, b = to_gray255(fused), to_gray255(src_a), to_gray255(src_b)
    refs = [to_gray255(ref)] if ref is not None else [a, b]
    mean = lambda fn: float(np.mean([fn(r, f) for r in refs]))
    return {
        "Qabf": qabf(a, b, f),
        "SSIM": mean(ssim_np),
        "VIF": mean(vifp),
        "EN": entropy(f),
        "FMI": feature_mutual_information(f, a, b),
        "AG": avg_gradient(f),
        "SF": spatial_frequency(f),
        "MI": mutual_information(f, a, b),
        "PSNR": mean(psnr),
        "RMSE": mean(rmse),
    }
