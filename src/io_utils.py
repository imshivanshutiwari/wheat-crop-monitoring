"""Local data ingestion: Resourcesat-2 AWiFS/LISS-III GeoTIFFs and
IMD gridded NetCDF rainfall/temperature."""
import numpy as np


def load_awifs_geotiff(path, bands=None):
    """Load an AWiFS/LISS-III GeoTIFF → (array, profile).

    NOTE: AWiFS is 56 m resolution — expect mixed-pixel anomalies on
    smallholder plots (<0.5 ha). Use Sentinel-2 (10-20 m) for
    plot-level mapping and AWiFS for wide-swath state coverage.
    """
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(bands) if bands else src.read()
        profile = src.profile
    return arr, profile


def awifs_ndvi(red, nir):
    """NDVI from AWiFS band arrays (band 2=red, band 3=NIR)."""
    red = red.astype("float32")
    nir = nir.astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    return np.clip(np.nan_to_num(ndvi), -1, 1)


def load_imd_netcdf(path, var=None):
    """Load IMD gridded NetCDF (rainfall 0.25°, temperature 1°) → xarray.

    IMD files use coordinates (TIME, LATITUDE, LONGITUDE); variable
    names vary (e.g. 'RAINFALL', 'TMAX'). Pass var to select one.
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    return ds[var] if var else ds


def imd_state_mean(da, state_gdf, lat_name="LATITUDE", lon_name="LONGITUDE"):
    """Mean of an IMD DataArray within a state polygon (bbox approx +
    point-in-polygon refinement via geopandas)."""
    import geopandas as gpd
    from shapely.geometry import Point
    minx, miny, maxx, maxy = state_gdf.total_bounds
    sub = da.sel({lat_name: slice(miny, maxy), lon_name: slice(minx, maxx)})
    lats = sub[lat_name].values
    lons = sub[lon_name].values
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for y in lats for x in lons],
                           crs=state_gdf.crs)
    inside = gpd.sjoin(pts, state_gdf, predicate="within")
    if inside.empty:
        return float(sub.mean())
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for geom in inside.geometry:
        iy = int(np.argmin(np.abs(lats - geom.y)))
        ix = int(np.argmin(np.abs(lons - geom.x)))
        mask[iy, ix] = True
    return float(sub.where(mask).mean())
