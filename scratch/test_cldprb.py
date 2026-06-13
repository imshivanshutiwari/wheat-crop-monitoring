import sys, yaml
sys.path.append('.')
import ee
from src import gee_utils

print("Initializing Earth Engine...")
gee_utils.init_ee()

aoi = gee_utils.get_state_geometry('Punjab')
start = ee.Date.fromYMD(2024, 1, 15)
end = ee.Date.fromYMD(2024, 2, 28)

print("Fetching S2 using MSK_CLDPRB band (no Join)...")
s2_fast = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(start, end))

def mask_clouds_fast(img):
    # MSK_CLDPRB is the cloud probability band in Sentinel-2 SR
    prob = img.select('MSK_CLDPRB')
    return img.updateMask(prob.lt(40)).divide(10000).copyProperties(img, ["system:time_start"])

s2_masked = s2_fast.map(mask_clouds_fast)
# Let's try getting the size or a simple value
print("Collection size:", s2_masked.size().getInfo())
