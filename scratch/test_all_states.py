import sys, yaml
sys.path.append('.')
import ee
import pandas as pd
import concurrent.futures
from src import gee_utils, indices, classification

with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)

print("Initializing Earth Engine...")
gee_utils.init_ee()
SEASON_YEAR = 2023
MONTHS = cfg['classification']['monthly_composites']

state = 'Punjab'
print(f"Loading state geometry for training on {state}...")
aoi_state = gee_utils.get_state_geometry(state)

print("Building feature stack for training...")
stack = classification.build_feature_stack(aoi_state, SEASON_YEAR, MONTHS, gee_utils, indices)

print("Generating stratified training points...")
peak_ndvi = (gee_utils.get_s2_collection(aoi_state, f'{SEASON_YEAR + 1}-01-15', f'{SEASON_YEAR + 1}-02-28')
             .map(indices.add_ndvi_s2).select('NDVI').median())
label_img = peak_ndvi.gt(0.55).rename('class')
samples_fc = label_img.stratifiedSample(numPoints=300, classBand='class', region=aoi_state,
                                        scale=250, seed=42, geometries=True).randomColumn(seed=1)
train_fc = samples_fc.filter(ee.Filter.lt('random', cfg['classification']['train_split']))

print("Training Random Forest classifier...")
clf, _ = classification.train_rf(stack, train_fc, 'class', cfg['classification']['n_trees'], scale=250)

def process_state(st):
    print(f"Processing {st}...")
    try:
        g = gee_utils.get_state_geometry(st)
        stk = classification.build_feature_stack(g, SEASON_YEAR, MONTHS, gee_utils, indices)
        wm = classification.classify(stk, clf)
        # Using scale=1000 to prevent Earth Engine timeout on large states
        area = classification.area_lakh_ha(wm, g, scale=1000).getInfo()
        print(f"-> {st} completed: {area:.2f} lakh ha")
        return {'state': st, 'area_lakh_ha': area}
    except Exception as e:
        print(f"-> {st} failed: {e}")
        return {'state': st, 'area_lakh_ha': None}

print("Starting concurrent GEE classifications for all states with scale=1000...")
with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['states'])) as executor:
    rows = list(executor.map(process_state, cfg['states']))

sown_df = pd.DataFrame(rows)
print("\nResults:")
print(sown_df)
