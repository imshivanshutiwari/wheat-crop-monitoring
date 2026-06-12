import json

notebook_path = r"e:\codes\Gitlab project\wheat-crop-monitoring\notebooks\03_vhi_crop_health_monitoring.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The new optimized cell source
new_source = """start = ee.Date(f'{SEASON_YEAR}-11-01')
n_fortnights = 12  # Nov -> Apr

# Define list of fortnight indices
fortnight_indices = ee.List.sequence(0, n_fortnights - 1)

# Map over fortnights to get a FeatureCollection per fortnight
def calculate_fortnight_vhi(i):
    i = ee.Number(i)
    f0 = start.advance(i.multiply(15), 'day')
    f1 = f0.advance(15, 'day')
    
    ndvi_now = gee_utils.get_modis_ndvi(aoi, f0, f1).mean()
    lst_now = gee_utils.get_modis_lst(aoi, f0, f1).mean()
    
    vci = indices.compute_vci(ndvi_now, ndvi_clim)
    tci = indices.compute_tci(lst_now, lst_clim)
    vhi = indices.compute_vhi(vci, tci, VHI_CFG['alpha'])
    
    # Spatial reduction: mean VHI for all states
    state_means = vhi.reduceRegions(
        collection=states_fc,
        reducer=ee.Reducer.mean(),
        scale=1000
    )
    
    date_str = f0.format('YYYY-MM-dd')
    
    # Map over states to add date and clean name
    def format_feature(f):
        return f.set({
            'date': date_str,
            'state': f.get('ADM1_NAME'),
            'VHI': f.get('VHI')
        })
    
    return state_means.map(format_feature)

# Flatten list of feature collections into a single feature collection
results_fc = ee.FeatureCollection(fortnight_indices.map(calculate_fortnight_vhi)).flatten()

# Pull all records in one single getInfo() call
results = results_fc.select(['date', 'state', 'VHI']).getInfo()

# Build Pandas DataFrame
rows = [feat['properties'] for feat in results['features']]
vhi_df = pd.DataFrame(rows)
vhi_df['date'] = pd.to_datetime(vhi_df['date'])
vhi_df.to_csv('../outputs/vhi_fortnightly.csv', index=False)
vhi_df.head()"""

# Replace cell 4's source (the index is 4 as we verified)
# Standard ipynb json expects source to be either a string or a list of strings.
# A string is clean and standard for modern nbformat.
nb["cells"][4]["source"] = new_source

# Save notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 03 successfully optimized!")
