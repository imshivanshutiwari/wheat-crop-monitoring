"""Test: run the exact patched NB04 cell-1 logic end-to-end."""
import pandas as pd, numpy as np

TARGET_STATES = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh',
                 'Rajasthan', 'Bihar', 'Gujarat', 'Maharashtra']

raw = pd.read_csv(r'e:\codes\Gitlab project\wheat-crop-monitoring\data\crop-wise-area-production-yield.csv')
wheat = raw[raw['crop_name'].str.strip().str.lower() == 'wheat'].copy()
wheat = wheat[wheat['state_name'].isin(TARGET_STATES)]
# Wheat is a Rabi crop in India
wheat = wheat[wheat['season'].str.strip().str.lower() == 'rabi']

wheat['year_parsed'] = wheat['year'].str.split('-').str[1].astype(int)
wheat['sown_area_kha'] = wheat['area'] / 1000
wheat = wheat.drop(columns=['year'])

hist = wheat.rename(columns={
    'state_name': 'state',
    'district_name': 'district',
    'year_parsed': 'year',
    'yield': 'yield_t_ha',
})[['state', 'district', 'year', 'yield_t_ha', 'sown_area_kha']]

hist = hist.dropna(subset=['yield_t_ha', 'sown_area_kha'])
hist = hist[hist['yield_t_ha'] > 0]
hist = hist.sort_values(['state', 'district', 'year']).reset_index(drop=True)

n = len(hist)
s = hist["state"].nunique()
d = hist["district"].nunique()
ymin = hist["year"].min()
ymax = hist["year"].max()
print(f'OK: {n} records | {s} states | {d} districts')
print(f'Year range: {ymin} - {ymax}')
print(hist.head(10).to_string())
