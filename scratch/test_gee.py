import sys, os, ee
import pandas as pd
import numpy as np

ee.Initialize(project="wheat-crop-monitoring-499208")

# Load real history to get district list
TARGET_STATES = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh',
                 'Rajasthan', 'Bihar', 'Gujarat', 'Maharashtra']

raw = pd.read_csv(r'e:\codes\Gitlab project\wheat-crop-monitoring\data\crop-wise-area-production-yield.csv')
wheat = raw[raw['crop_name'].str.strip().str.lower() == 'wheat']
wheat = wheat[wheat['state_name'].isin(TARGET_STATES)]
wheat = wheat[wheat['season'].str.strip().str.lower() == 'rabi']
wheat['year_parsed'] = wheat['year'].str.split('-').str[1].astype(int)
wheat['sown_area_kha'] = wheat['area'] / 1000
wheat = wheat.drop(columns=['year'])
hist = wheat.rename(columns={'state_name': 'state', 'district_name': 'district', 'year_parsed': 'year', 'yield': 'yield_t_ha'})
hist = hist[['state', 'district', 'year', 'yield_t_ha', 'sown_area_kha']].dropna()
hist = hist[hist['yield_t_ha'] > 0].reset_index(drop=True)

# GEE District Geometries
gaul = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "India"))
gaul_states = ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Bihar", "Gujarat", "Maharashtra"]
gaul = gaul.filter(ee.Filter.inList("ADM1_NAME", gaul_states))

# Test for 1 year
test_year = 2020
print(f"Extracting for Rabi season {test_year-1}-{test_year}")

start_date = f"{test_year-1}-11-01"
end_date = f"{test_year}-04-30"

# MODIS NDVI
modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date).select("NDVI")
# Get mean NDVI over the season
mean_ndvi = modis.mean().multiply(0.0001)

# Reduce regions
reduced = mean_ndvi.reduceRegions(
    collection=gaul,
    reducer=ee.Reducer.mean(),
    scale=5000 # 5km scale for speed
)

print("Fetching from GEE...")
features = reduced.getInfo()['features']
print(f"Fetched {len(features)} districts")

rows = []
for f in features:
    props = f['properties']
    if 'mean' in props:
        rows.append({
            'state': props.get('ADM1_NAME'),
            'district': props.get('ADM2_NAME'),
            'year': test_year,
            'ndvi_mean': props.get('mean')
        })

df = pd.DataFrame(rows)
print(df.head())
