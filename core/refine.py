"""Colour anchoring for the refine pass.

The refine pass re-generates every view from one shared texture. It removes most view-to-view disagreement and the
silhouette fringe, but the VAE round trip can move the local average colour (measured: the iris and body drift warmer
and darker). The refined view keeps all of its own detail; only its local average colour is pulled back to the guide
render it started from, with a gain that is smooth over `sigma` pixels and bounded by `limit`.
"""
import numpy as np
from scipy import ndimage as ndi

SCALE = 4   # the smooth gain is computed on a 1/SCALE grid


def _to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_srgb(linear):
    linear = np.clip(linear, 0, None)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055).astype(np.float32)


def _down(a):
    h, w = a.shape[:2]
    return a[:h // SCALE * SCALE, :w // SCALE * SCALE].reshape(h // SCALE, SCALE, w // SCALE, SCALE, *a.shape[2:]).mean((1, 3))


def match_low_frequency(image, guide, mask, sigma=12.0, limit=0.30):
    """image: uint8 (S,S,3) refined view; guide: float (S,S,3) sRGB in 0..1; mask: bool (S,S). Returns uint8.

    Pixels outside `mask` are returned unchanged. The gain is the ratio of the mask-normalised Gaussian means of the
    guide and of the refined view in linear light, clipped to 1 +- limit.
    """
    if image.shape != guide.shape or image.shape[:2] != mask.shape or image.shape[0] % SCALE:
        raise ValueError("image, guide and mask must be aligned square arrays with a side divisible by %d" % SCALE)
    refined = _to_linear(image.astype(np.float32) / 255.0)
    target = _to_linear(np.clip(guide, 0, 1))
    weight = mask.astype(np.float32)
    ws = ndi.gaussian_filter(_down(weight), sigma / SCALE)

    def smooth(a):
        return ndi.gaussian_filter(_down(a * weight[..., None]), (sigma / SCALE, sigma / SCALE, 0)) / np.maximum(ws, 1e-3)[..., None]

    gain = (smooth(target) + 1e-3) / (smooth(refined) + 1e-3)
    gain = np.clip(gain, 1 - limit, 1 + limit)
    full = ndi.zoom(gain, (SCALE, SCALE, 1), order=1, mode="nearest")
    out = np.where(mask[..., None], _to_srgb(refined * full), image.astype(np.float32) / 255.0)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
