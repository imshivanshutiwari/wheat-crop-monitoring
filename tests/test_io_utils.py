import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from src.io_utils import awifs_ndvi, imd_state_mean


def test_awifs_ndvi():
    red = np.array([0.1, 0.2, 0.3])
    nir = np.array([0.2, 0.3, 0.4])
    res = awifs_ndvi(red, nir)
    expected = (nir - red) / (nir + red)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_imd_state_mean_ascending():
    # Ascending coordinates
    lats = np.array([10.0, 11.0, 12.0])
    lons = np.array([70.0, 71.0, 72.0])
    data = np.arange(9).reshape(3, 3)
    da = xr.DataArray(data, coords=[("LATITUDE", lats), ("LONGITUDE", lons)])

    # State geometry: box from 70.5 to 72.5 lon, 10.5 to 12.5 lat
    poly = Polygon([(70.5, 10.5), (72.5, 10.5), (72.5, 12.5), (70.5, 12.5)])
    state_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    val = imd_state_mean(da, state_gdf)
    # Inside cells will be: (11.0, 71.0) -> value 4, (11.0, 72.0) -> value 5
    # (12.0, 71.0) -> value 7, (12.0, 72.0) -> value 8
    # Average of 4, 5, 7, 8 is 6.0
    assert abs(val - 6.0) < 1e-5


def test_imd_state_mean_descending():
    # Descending coordinates
    lats = np.array([12.0, 11.0, 10.0])
    lons = np.array([72.0, 71.0, 70.0])
    data = np.arange(9).reshape(3, 3)
    da = xr.DataArray(data, coords=[("LATITUDE", lats), ("LONGITUDE", lons)])

    # State geometry: box from 70.5 to 72.5 lon, 10.5 to 12.5 lat
    poly = Polygon([(70.5, 10.5), (72.5, 10.5), (72.5, 12.5), (70.5, 12.5)])
    state_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    val = imd_state_mean(da, state_gdf)
    # Under slice(maxy, miny) / slice(maxx, minx):
    # lat_slice = slice(12.5, 10.5), lon_slice = slice(72.5, 70.5)
    # Lats selected: 12.0, 11.0. Lons selected: 72.0, 71.0.
    # Inside cells in sub: (12, 72) -> 0, (12, 71) -> 1, (11, 72) -> 3, (11, 71) -> 4
    # Mean of 0, 1, 3, 4 is 2.0
    assert abs(val - 2.0) < 1e-5
