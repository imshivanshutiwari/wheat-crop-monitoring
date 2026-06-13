import sys, yaml
sys.path.append('.')
import ee
from src import gee_utils, indices, classification

with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)

print("Initializing Earth Engine...")
gee_utils.init_ee()
SEASON_YEAR = 2023
MONTHS = cfg['classification']['monthly_composites']

state = 'Punjab'
print(f"Loading state geometry for {state}...")
aoi_state = gee_utils.get_state_geometry(state)

print("Building feature stack...")
stack = classification.build_feature_stack(aoi_state, SEASON_YEAR, MONTHS, gee_utils, indices)

print("Generating pseudo-labels via fast randomPoints + sampleRegions...")
peak_ndvi = (gee_utils.get_s2_collection(aoi_state, f'{SEASON_YEAR + 1}-01-15', f'{SEASON_YEAR + 1}-02-28')
             .map(indices.add_ndvi_s2).select('NDVI').median())
label_img = peak_ndvi.gt(0.55).rename('class')

# instantaneous geometric generation
random_points = ee.FeatureCollection.randomPoints(aoi_state, 600, seed=42)
# sample the image only at those 600 points
samples_fc = label_img.sampleRegions(
    collection=random_points,
    properties=['class'],
    scale=30,
    geometries=True
).randomColumn(seed=1)

train_fc = samples_fc.filter(ee.Filter.lt('random', cfg['classification']['train_split']))
val_fc = samples_fc.filter(ee.Filter.gte('random', cfg['classification']['train_split']))
print('Train size:', train_fc.size().getInfo(), 'Val size:', val_fc.size().getInfo())

print("Training Random Forest classifier...")
clf, _ = classification.train_rf(stack, train_fc, 'class', cfg['classification']['n_trees'], scale=250)

print("Evaluating validation samples...")
val_samples = stack.sampleRegions(collection=val_fc, properties=['class'], scale=250, tileScale=4)
report = classification.accuracy_report(clf, val_samples, 'class')
print('Overall accuracy:', round(report['overall_accuracy'], 3), '| kappa:', round(report['kappa'], 3))

print("Classifying wheat map and calculating area...")
wheat_map = classification.classify(stack, clf)
area = classification.area_lakh_ha(wheat_map, aoi_state, scale=250).getInfo()
print(f'{state} wheat sown area: {area:.2f} lakh ha')
