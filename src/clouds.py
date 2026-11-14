"""Simulated cloud cover for robustness tests.

apply_clouds(optical, cover, seed) returns a copy of optical (N, C, H, W) with NaN over smooth random blobs that
cover exactly `cover` of each patch's pixels (cover 0.0 -> unchanged, 1.0 -> all NaN). The blob shape for patch i
depends only on (seed, i), so every model sees the same clouds.
"""
import numpy as np


def _blob_field(h, w, rng, scale=32):
    # low-frequency noise: coarse random grid upsampled with bilinear interpolation
    gh, gw = h // scale + 2, w // scale + 2
    g = rng.standard_normal((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx


def cloud_mask(h, w, cover, seed, index):
    """Boolean (H, W) mask, True = cloud, with round(cover * H * W) cloudy pixels."""
    if cover <= 0:
        return np.zeros((h, w), bool)
    if cover >= 1:
        return np.ones((h, w), bool)
    rng = np.random.default_rng([seed, index])
    field = _blob_field(h, w, rng)
    k = int(round(cover * h * w))
    thresh = np.partition(field.ravel(), -k)[-k]
    return field >= thresh


def apply_clouds(optical, cover, seed, start_index=0):
    """optical: (N, C, H, W) float array. Returns a float32 copy with NaN under the cloud mask."""
    out = np.array(optical, dtype=np.float32, copy=True)
    n, _, h, w = out.shape
    for i in range(n):
        m = cloud_mask(h, w, cover, seed, start_index + i)
        out[i][:, m] = np.nan
    return out
