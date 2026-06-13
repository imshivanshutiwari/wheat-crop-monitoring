import json

notebook_path = r"notebooks/02_sar_optical_fusion_sown_area.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

modified_cells = 0

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        new_source = []
        changed = False
        for line in source:
            # Cell 5: single state area calculation
            if "classification.area_lakh_ha(wheat_map, aoi_state)" in line and "scale=" not in line:
                line = line.replace(
                    "classification.area_lakh_ha(wheat_map, aoi_state)",
                    "classification.area_lakh_ha(wheat_map, aoi_state, scale=1000)"
                )
                changed = True
            
            # Cell 6: thread pool area calculation
            if "classification.area_lakh_ha(wm, g)" in line and "scale=" not in line:
                line = line.replace(
                    "classification.area_lakh_ha(wm, g)",
                    "classification.area_lakh_ha(wm, g, scale=1000)"
                )
                changed = True
            
            # Cell 7: unused wheat_zone intersection
            if "wheat_zone = aoi_state.intersection" in line and not line.strip().startswith("#"):
                line = "# " + line + " # Unused and causes GEE timeout\n"
                changed = True
            
            new_source.append(line)
        
        if changed:
            cell["source"] = new_source
            modified_cells += 1

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Success: Modified {modified_cells} cells in {notebook_path}")
