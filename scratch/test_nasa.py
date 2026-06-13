import os, pandas as pd
w_dir = r'e:\codes\Gitlab project\wheat-crop-monitoring\data\nasa_power'
if os.path.exists(w_dir):
    w_files = [f for f in os.listdir(w_dir) if f.endswith('.csv')]
    if w_files:
        df = pd.read_csv(os.path.join(w_dir, w_files[0]), comment='#')
        print(f'{len(w_files)} files found. Columns in first: {list(df.columns)}')
        print(f'Years covered: {df["YEAR"].min()} - {df["YEAR"].max()}')
    else:
        print('No csv files')
else:
    print('Dir not found')
