import pandas as pd
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Given" / "compas.csv"
df = pd.read_csv(DATAPATH)
print("\nInformation: ")
df.info()
print("\n Statistics: ")
print(df.describe())
print("\n Number of Null Values: ")
print(df.isna().sum())
print("\nShape: ")
print(df.shape)
print("\nValue Counts")
for i in df.columns:
    print(f"{i}: ")
    print(df[i].value_counts())
print(df.columns.tolist())
