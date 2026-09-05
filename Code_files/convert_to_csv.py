import pandas as pd
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "german_credit_data.xlsx"
df = pd.read_excel(DATAPATH)
df.to_csv(BASEDIR / "datasets" / "Mine" / "german_credit_data.csv", index=False)