"""Linear spectral unmixing for AWiFS 56 m mixed pixels.

Smallholder plots are sub-pixel at 56 m, so a hard classification
mislabels mixed pixels. Fully-constrained least squares unmixing
(FCLS) estimates the *fraction* of wheat within each coarse pixel,
enabling sub-pixel sown-area estimation.

Reference: Heinz & Chang (2001), FCLS.
"""
import numpy as np
from scipy.optimize import nnls


def fcls_unmix(pixels, endmembers):
    """Fully-constrained (sum-to-one, non-negative) linear unmixing.

    Args:
        pixels: (N, B) reflectance spectra.
        endmembers: (E, B) pure class spectra (e.g. wheat, soil, other crop).
    Returns:
        (N, E) abundance fractions, each row summing to ~1.
    """
    pixels = np.atleast_2d(np.asarray(pixels, float))
    E = np.asarray(endmembers, float)
    n, _ = pixels.shape
    n_e = E.shape[0]
    delta = 1e4  # sum-to-one weight (ASC via augmentation)
    A = np.vstack([E.T, np.full((1, n_e), delta)])
    out = np.zeros((n, n_e))
    for k in range(n):
        b = np.concatenate([pixels[k], [delta]])
        sol, _ = nnls(A, b)
        s = sol.sum()
        out[k] = sol / s if s > 0 else sol
    return out


def wheat_fraction_area(abundances, wheat_idx, pixel_area_ha):
    """Sub-pixel wheat area (ha) from abundance map.

    abundances: (N, E) or (H, W, E); pixel_area_ha for a 56 m pixel
    (~0.3136 ha). Returns total wheat area in hectares.
    """
    a = np.asarray(abundances)
    frac = a[..., wheat_idx]
    return float(np.nansum(frac) * pixel_area_ha)


def extract_endmembers_ppi(cube, n_classes=3, seed=0):
    """Naive endmember seeds via spectral extremes (k-means style init).

    For production use VCA/PPI; this gives a reproducible starting set.
    cube: (B, H, W). Returns (n_classes, B).
    """
    B, H, W = cube.shape
    flat = cube.reshape(B, -1).T
    flat = flat[np.isfinite(flat).all(1)]
    rng = np.random.default_rng(seed)
    brightness = flat.sum(1)
    idx = [int(np.argmin(brightness)), int(np.argmax(brightness))]
    while len(idx) < n_classes:
        idx.append(int(rng.integers(0, flat.shape[0])))
    return flat[idx]
