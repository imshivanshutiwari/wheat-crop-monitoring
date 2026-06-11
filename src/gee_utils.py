"""Google Earth Engine helpers: init, AOIs, cloud masking, composites."""
import ee

GAUL_STATES = "FAO/GAUL/2015/level1"


def init_ee(project=None):
    """Initialize Earth Engine (run ee.Authenticate() once beforehand)."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def get_state_geometry(state_name, country="India"):
    """Return ee.Geometry for an Indian state from FAO GAUL level-1."""
    fc = (ee.FeatureCollection(GAUL_STATES)
          .filter(ee.Filter.eq("ADM0_NAME", country))
          .filter(ee.Filter.eq("ADM1_NAME", state_name)))
    return fc.geometry()


def get_states_fc(state_names, country="India"):
    """FeatureCollection of multiple states."""
    return (ee.FeatureCollection(GAUL_STATES)
            .filter(ee.Filter.eq("ADM0_NAME", country))
            .filter(ee.Filter.inList("ADM1_NAME", list(state_names))))


def mask_s2_clouds(img, max_cloud_prob=40):
    """Mask Sentinel-2 SR image using joined s2cloudless probability."""
    prob = ee.Image(img.get("cloud_prob")).select("probability")
    return img.updateMask(prob.lt(max_cloud_prob)).divide(10000) \
              .copyProperties(img, ["system:time_start"])


def get_s2_collection(aoi, start, end, max_cloud_prob=40):
    """Cloud-masked Sentinel-2 SR collection (joined with s2cloudless)."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end))
    clouds = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
              .filterBounds(aoi).filterDate(start, end))
    joined = ee.Join.saveFirst("cloud_prob").apply(
        primary=s2, secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index",
                                   rightField="system:index"))
    return ee.ImageCollection(joined).map(
        lambda img: mask_s2_clouds(ee.Image(img), max_cloud_prob))


def get_s1_collection(aoi, start, end, orbit="DESCENDING"):
    """Sentinel-1 GRD IW VV+VH collection in dB."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]))


def monthly_composite(collection, year, month, reducer=None):
    """Median (or custom) monthly composite."""
    reducer = reducer or ee.Reducer.median()
    start = ee.Date.fromYMD(year, month, 1)
    return (collection.filterDate(start, start.advance(1, "month"))
            .reduce(reducer)
            .set("year", year).set("month", month,))


def get_modis_ndvi(aoi, start, end):
    """MODIS MOD13Q1 NDVI scaled to [-1, 1]."""
    return (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterBounds(aoi).filterDate(start, end)
            .select("NDVI")
            .map(lambda i: i.multiply(0.0001)
                 .copyProperties(i, ["system:time_start"])))


def get_modis_lst(aoi, start, end):
    """MODIS MOD11A2 day LST in Celsius."""
    return (ee.ImageCollection("MODIS/061/MOD11A2")
            .filterBounds(aoi).filterDate(start, end)
            .select("LST_Day_1km")
            .map(lambda i: i.multiply(0.02).subtract(273.15)
                 .copyProperties(i, ["system:time_start"])))
