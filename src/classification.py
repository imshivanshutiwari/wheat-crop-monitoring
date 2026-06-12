"""Wheat sown-area mapping: SAR-optical feature fusion + Random Forest (GEE)."""
import ee


def build_feature_stack(aoi, season_year, months, gee_utils, indices):
    """Monthly S1 (VV, VH) + S2 NDVI composites stacked into one image.

    season_year: calendar year in which the Rabi season *starts* (November).
    months 11,12 belong to season_year; 1..4 to season_year+1.
    """
    bands = []
    for m in months:
        yr = season_year if m >= 11 else season_year + 1
        start = ee.Date.fromYMD(yr, m, 1)
        end = start.advance(1, "month")
        s1 = gee_utils.get_s1_collection(aoi, start, end).median() \
            .select(["VV", "VH"]).rename([f"VV_{m:02d}", f"VH_{m:02d}"])
        s2 = gee_utils.get_s2_collection(aoi, start, end) \
            .map(indices.add_ndvi_s2).select("NDVI").median() \
            .rename(f"NDVI_{m:02d}")
        bands.append(s1)
        bands.append(s2)
    return ee.Image.cat(bands).clip(aoi)


def generate_sample_training_points(aoi, n_per_class=300, seed=42):
    """DEMO ONLY: pseudo-labels from a winter NDVI heuristic.

    Wheat shows high NDVI in Jan-Feb. Replace with real ground-truth
    points (crop cutting experiment / field survey shapefiles) for
    operational use.
    """
    # Heuristic label image must be provided by caller in practice;
    # this samples stratified random points inside the AOI.
    return ee.FeatureCollection.randomPoints(aoi, n_per_class * 2, seed)


def train_rf(feature_stack, training_fc, label_prop="class",
             n_trees=200, scale=20):
    """Sample feature stack at training points and fit smileRandomForest."""
    samples = feature_stack.sampleRegions(
        collection=training_fc, properties=[label_prop],
        scale=scale, tileScale=4, geometries=False)
    classifier = ee.Classifier.smileRandomForest(n_trees) \
        .train(samples, label_prop, feature_stack.bandNames())
    return classifier, samples


def classify(feature_stack, classifier):
    """Apply trained classifier → wheat (1) / non-wheat (0) map."""
    return feature_stack.classify(classifier).rename("wheat")


def area_lakh_ha(wheat_map, aoi, scale=250):
    """Wheat area in lakh hectares (1 lakh ha = 1e5 ha = 1e9 m2)."""
    area_img = wheat_map.eq(1).multiply(ee.Image.pixelArea())
    m2 = area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi,
        scale=scale, maxPixels=1e12, bestEffort=True).get("wheat")
    return ee.Number(m2).divide(1e9)  # m2 → lakh ha


def accuracy_report(classifier, validation_samples, label_prop="class"):
    """Confusion matrix + overall accuracy + kappa on held-out samples."""
    validated = validation_samples.classify(classifier)
    cm = validated.errorMatrix(label_prop, "classification")
    return {
        "confusion_matrix": cm.getInfo(),
        "overall_accuracy": cm.accuracy().getInfo(),
        "kappa": cm.kappa().getInfo(),
    }
