"""Optional preprocessing (paper Sec. 3.3): ECC registration + luminance histogram equalisation.
OpenCV is imported lazily so the rest of the package works without it."""
import numpy as np


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    """img: HxWx3 uint8 RGB. Equalises only the luminance (Y of YCrCb) to avoid colour shifts."""
    import cv2
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def register_ecc(moving: np.ndarray, reference: np.ndarray, iters: int = 100, eps: float = 1e-5) -> np.ndarray:
    """Align `moving` to `reference` with an affine ECC model. Returns `moving` unchanged on failure."""
    import cv2
    g_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_mov = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp = cv2.findTransformECC(g_ref, g_mov, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return moving
    h, w = reference.shape[:2]
    return cv2.warpAffine(moving, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_pair(a: np.ndarray, b: np.ndarray, register: bool = False, hist_eq: bool = False):
    """a, b: HxWx3 uint8 RGB arrays of identical size."""
    if register:
        b = register_ecc(b, a)
    if hist_eq:
        a, b = histogram_equalize(a), histogram_equalize(b)
    return a, b
