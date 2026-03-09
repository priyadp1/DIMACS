import pandas as pd
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "bike.csv"
df = pd.read_csv(DATAPATH)
print("\nInformation: ")
df.info()
print("\n Statistics: ")
print(df.describe())
print("\n Number of Null Values: ")
print(df.isna().sum())
print("\nShape: ")
print(df.shape)
print(df.columns.tolist())
print(df["label"].value_counts())
