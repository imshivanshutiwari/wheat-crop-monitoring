import json

notebook_path = r"e:\codes\Gitlab project\wheat-crop-monitoring\notebooks\02_sar_optical_fusion_sown_area.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The new optimized cell source
new_source = """import concurrent.futures

def process_state(st):
    g = gee_utils.get_state_geometry(st)
    stk = classification.build_feature_stack(g, SEASON_YEAR, MONTHS, gee_utils, indices)
    wm = classification.classify(stk, clf)  # operational: retrain per agro-climatic zone
    area = classification.area_lakh_ha(wm, g).getInfo()
    return {'state': st, 'area_lakh_ha': area}

# Run GEE classifications in parallel using ThreadPoolExecutor
print("Starting concurrent GEE classifications for all states...")
with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['states'])) as executor:
    rows = list(executor.map(process_state, cfg['states']))

sown_df = pd.DataFrame(rows)
sown_df.to_csv('../outputs/sown_area_estimates.csv', index=False)
print('Total:', round(sown_df.area_lakh_ha.sum(), 1), 'lakh ha')"""

# Find the cell with "rows = []" and "for st in cfg['states']:"
found = False
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "rows = []" in "".join(cell["source"]):
        nb["cells"][idx]["source"] = new_source
        found = True
        print(f"Found and replaced cell at index {idx}")
        break

if not found:
    print("Could not find the target code cell!")
else:
    # Save notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook 02 successfully optimized!")
