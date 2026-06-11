"""Sentinel-1 SAR processing: speckle filtering, backscatter time-series."""
import ee
import pandas as pd


def speckle_filter(img, radius_m=50):
    """Focal-median speckle filter on VV/VH (dB)."""
    return (img.focal_median(radius_m, "circle", "meters")
            .copyProperties(img, ["system:time_start"]))


def add_ratio(img):
    """Add VH/VV ratio band (dB difference)."""
    return img.addBands(
        img.select("VH").subtract(img.select("VV")).rename("VH_VV"))


def backscatter_timeseries(s1_collection, aoi, band="VH", scale=100):
    """Mean backscatter over AOI per acquisition → pandas DataFrame.

    Used for SAR backscatter vs crop phenology curves.
    """
    def _reduce(img):
        stat = img.select(band).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=scale, maxPixels=1e10, bestEffort=True)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            band: stat.get(band)})

    feats = s1_collection.map(_reduce).filter(
        ee.Filter.notNull([band])).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in feats])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


WHEAT_PHENOLOGY = {
    # Approximate Rabi wheat growth stages (North India)
    "Sowing":    ("11-01", "11-30"),
    "Tillering": ("12-01", "12-31"),
    "Jointing":  ("01-01", "01-31"),
    "Heading":   ("02-01", "02-28"),
    "Maturity":  ("03-01", "03-31"),
    "Harvest":   ("04-01", "04-30"),
}
