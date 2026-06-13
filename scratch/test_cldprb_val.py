import sys, yaml
sys.path.append('.')
import ee
from src import gee_utils, indices

print("Initializing Earth Engine...")
gee_utils.init_ee()

aoi = gee_utils.get_state_geometry('Punjab')
start = ee.Date.fromYMD(2024, 1, 15)
end = ee.Date.fromYMD(2024, 2, 28)

def mask_clouds_fast(img):
    prob = img.select('MSK_CLDPRB')
    # Since we divide by 10000 to scale bands to [0, 1] (except probability band)
    # let's make sure we update the mask first
    img_masked = img.updateMask(prob.lt(40))
    # Scale reflectance bands (B1-B12) to [0, 1]
    # In the original mask_s2_clouds:
    # img.updateMask(prob.lt(max_cloud_prob)).divide(10000).copyProperties(...)
    # Note: dividing the whole image by 10000 also divides QA/MSK bands, but we only need spectral bands.
    # To keep compatibility, let's just divide the whole image by 10000.
    return img_masked.divide(10000).copyProperties(img, ["system:time_start"])

print("Building NDVI composite...")
s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end)
          .map(mask_clouds_fast))

peak_ndvi = s2_col.map(indices.add_ndvi_s2).select('NDVI').median()

# Let's get the mean NDVI value over a small point in Punjab to verify it works and has non-null values
point = ee.Geometry.Point([75.8573, 30.9010]) # Ludhiana, Punjab
ndvi_val = peak_ndvi.reduceRegion(ee.Reducer.mean(), point, 30).get('NDVI').getInfo()
print("NDVI value at Ludhiana:", ndvi_val)
