"""Vegetation indices: NDVI, VCI, TCI, VHI (GEE server-side)."""
import ee


def add_ndvi_s2(img):
    """Add NDVI band to a Sentinel-2 SR image."""
    return img.addBands(
        img.normalizedDifference(["B8", "B4"]).rename("NDVI"))


def _minmax(collection, band, aoi, scale):
    mn = collection.select(band).min()
    mx = collection.select(band).max()
    return mn, mx


def compute_vci(ndvi_current, ndvi_climatology):
    """VCI = (NDVI - NDVImin) / (NDVImax - NDVImin) * 100.

    ndvi_climatology: multi-year ee.ImageCollection of NDVI.
    """
    ndvi_min = ndvi_climatology.min()
    ndvi_max = ndvi_climatology.max()
    return (ndvi_current.subtract(ndvi_min)
            .divide(ndvi_max.subtract(ndvi_min))
            .multiply(100).clamp(0, 100).rename("VCI"))


def compute_tci(lst_current, lst_climatology):
    """TCI = (LSTmax - LST) / (LSTmax - LSTmin) * 100."""
    lst_min = lst_climatology.min()
    lst_max = lst_climatology.max()
    return (lst_max.subtract(lst_current)
            .divide(lst_max.subtract(lst_min))
            .multiply(100).clamp(0, 100).rename("TCI"))


def compute_vhi(vci, tci, alpha=0.5):
    """VHI = alpha*VCI + (1-alpha)*TCI."""
    return vci.multiply(alpha).add(tci.multiply(1 - alpha)).rename("VHI")


def vhi_stress_mask(vhi, threshold=40):
    """Binary stress mask (VHI < threshold)."""
    return vhi.lt(threshold).rename("stressed")


def regional_mean(img, aoi, scale=1000):
    """Mean of all bands of img over aoi → ee.Dictionary."""
    return img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi,
        scale=scale, maxPixels=1e10, bestEffort=True)
