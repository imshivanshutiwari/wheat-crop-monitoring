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

    minx, miny, maxx, maxy = state_gdf.total_bounds

    # Handle potentially descending latitude or longitude coordinates
    lat_coords = da[lat_name].values
    if len(lat_coords) > 1 and lat_coords[0] > lat_coords[1]:
        lat_slice = slice(maxy, miny)
    else:
        lat_slice = slice(miny, maxy)

    lon_coords = da[lon_name].values
    if len(lon_coords) > 1 and lon_coords[0] > lon_coords[1]:
        lon_slice = slice(maxx, minx)
    else:
        lon_slice = slice(minx, maxx)

    sub = da.sel({lat_name: lat_slice, lon_name: lon_slice})
    lats = sub[lat_name].values
    lons = sub[lon_name].values

    if len(lats) == 0 or len(lons) == 0:
        return float(da.mean())

    # Create meshgrid of coordinates and flatten
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    # Vectorized construction of Point geometries
    points = gpd.points_from_xy(lon_flat, lat_flat)
    pts = gpd.GeoDataFrame(geometry=points, crs=state_gdf.crs)

    inside = gpd.sjoin(pts, state_gdf, predicate="within")
    if inside.empty:
        return float(sub.mean())

    # Vectorized mask construction from index of points within geometry
    mask_flat = np.zeros(len(points), dtype=bool)
    mask_flat[inside.index] = True
    mask = mask_flat.reshape(len(lats), len(lons))

    return float(sub.where(mask).mean())
