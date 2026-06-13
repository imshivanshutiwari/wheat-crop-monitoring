---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
~\AppData\Local\Temp\ipykernel_7968\1665150719.py in ?()
     29 })[['state', 'district', 'year', 'yield_t_ha', 'sown_area_kha']]
     30 
     31 hist = hist.dropna(subset=['yield_t_ha', 'sown_area_kha'])
     32 hist = hist[hist['yield_t_ha'] > 0]
---> 33 hist = hist.sort_values(['state', 'district', 'year']).reset_index(drop=True)
     34 print(f'✓ {len(hist)} district-year records | {hist["state"].nunique()} states | {hist["district"].nunique()} districts')
     35 print(f'  Year range: {hist["year"].min()} – {hist["year"].max()}')
     36 hist.head(10)

E:\codes\Gitlab project\wheat-crop-monitoring\venv\lib\site-packages\pandas\core\frame.py in ?(self, by, axis, ascending, inplace, kind, na_position, ignore_index, key)
   7190                 f"Length of ascending ({len(ascending)})"  # type: ignore[arg-type]
   7191                 f" != length of by ({len(by)})"
   7192             )
   7193         if len(by) > 1:
-> 7194             keys = [self._get_label_or_level_values(x, axis=axis) for x in by]
   7195 
   7196             # need to rewrap columns in Series to apply key function
   7197             if key is not None:

E:\codes\Gitlab project\wheat-crop-monitoring\venv\lib\site-packages\pandas\core\frame.py in ?(.0)
-> 7194         ...     key=lambda x: np.argsort(index_natsorted(df["time"]))

E:\codes\Gitlab project\wheat-crop-monitoring\venv\lib\site-packages\pandas\core\generic.py in ?(self, key, axis)
   1924             else:
   1925                 multi_message = ""
   1926 
   1927             label_axis_name = "column" if axis == 0 else "index"
-> 1928             raise ValueError(
   1929                 f"The {label_axis_name} label '{key}' is not unique.{multi_message}"
   1930             )
   1931 

ValueError: The column label 'year' is not unique.