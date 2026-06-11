"""Per-pixel phenology extraction via double-logistic curve fitting.

Fits a double-logistic to an NDVI (or VH backscatter) time-series and
derives phenological metrics: start of season (sowing), peak, end of
season (senescence), amplitude, and season length. Replaces the
heuristic fixed-window assumption with data-driven dates per pixel.

Reference: Beck et al. (2006); Zhang et al. (2003) logistic phenology.
"""
import numpy as np
from scipy.optimize import curve_fit


def double_logistic(t, vmin, vmax, sos, eos, slope_s, slope_e):
    """Double-logistic NDVI model.

    t: day-of-season array. sos/eos: start/end of season (days).
    """
    green_up = 1.0 / (1.0 + np.exp(-slope_s * (t - sos)))
    senescence = 1.0 / (1.0 + np.exp(slope_e * (t - eos)))
    return vmin + (vmax - vmin) * green_up * senescence


def fit_pixel(t, ndvi, p0=None, maxfev=5000):
    """Fit double-logistic to one pixel's NDVI series.

    Returns dict of phenology metrics or None if the fit fails.
    """
    t = np.asarray(t, float)
    ndvi = np.asarray(ndvi, float)
    ok = np.isfinite(ndvi)
    if ok.sum() < 6:
        return None
    t, ndvi = t[ok], ndvi[ok]
    if p0 is None:
        p0 = [float(np.nanmin(ndvi)), float(np.nanmax(ndvi)),
              t.min() + 0.25 * np.ptp(t), t.min() + 0.75 * np.ptp(t),
              0.1, 0.1]
    try:
        popt, _ = curve_fit(double_logistic, t, ndvi, p0=p0, maxfev=maxfev)
    except (RuntimeError, ValueError):
        return None
    vmin, vmax, sos, eos, ss, se = popt
    return {
        "sowing_doy": float(sos),
        "senescence_doy": float(eos),
        "peak_doy": float((sos + eos) / 2.0),
        "amplitude": float(vmax - vmin),
        "season_length": float(eos - sos),
        "params": popt,
    }


def fit_stack(t, cube):
    """Fit phenology over an (T, H, W) NDVI cube -> dict of metric maps."""
    T, H, W = cube.shape
    metrics = ["sowing_doy", "senescence_doy", "peak_doy",
               "amplitude", "season_length"]
    out = {m: np.full((H, W), np.nan, np.float32) for m in metrics}
    for i in range(H):
        for j in range(W):
            res = fit_pixel(t, cube[:, i, j])
            if res:
                for m in metrics:
                    out[m][i, j] = res[m]
    return out


def ndvi_integral(t, ndvi):
    """Integrated NDVI over the season (proxy for accumulated biomass)."""
    t = np.asarray(t, float)
    ndvi = np.clip(np.nan_to_num(np.asarray(ndvi, float)), 0, 1)
    return float(np.trapz(ndvi, t))
