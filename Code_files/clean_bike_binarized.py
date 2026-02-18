"""
Cleans bike_binarized.csv by sanitizing column names so they are
compatible with XGBoost (which forbids '<', '[', ']' in feature names).

Replaces ' <= ' with '_le_' in all column headers.
Output: datasets/Given/bike_binarized_new.csv
"""

import pandas as pd
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent
IN_PATH  = BASEDIR / "datasets" / "Given" / "bike_binarized.csv"
OUT_PATH = BASEDIR / "datasets" / "Given" / "bike_binarized_new.csv"

df = pd.read_csv(IN_PATH)

original_cols = df.columns.tolist()
df.columns = [col.replace(" <= ", "_le_") for col in original_cols]

df.to_csv(OUT_PATH, index=False)

print(f"Saved cleaned dataset to: {OUT_PATH}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("\nSample column renames:")
for old, new in zip(original_cols[:5], df.columns[:5]):
    if old != new:
        print(f"  '{old}'  ->  '{new}'")
