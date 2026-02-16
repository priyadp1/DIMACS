import kagglehub
import os
import shutil
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
# Download latest version
target = DATAPATH
os.makedirs(target, exist_ok=True)
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
for i in os.listdir(path):
    if i.endswith(".csv"):
        src = os.path.join(path,i)
        dest = os.path.join(target,i)
        shutil.copy(src, dest)
        print(f"Copied {i} to {target}")

print("Dataset successfully downloaded.")