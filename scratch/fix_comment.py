import json

notebook_path = r"notebooks/02_sar_optical_fusion_sown_area.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for idx, line in enumerate(source):
            if "wheat_zone = aoi_state.intersection" in line and "# Unused and causes GEE timeout" in line:
                source[idx] = "# wheat_zone = aoi_state.intersection(wheat_map.selfMask().geometry(), 1000) # Unused and causes GEE timeout\n"

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Comment successfully cleaned!")
