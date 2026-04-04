import kagglehub
import os
import shutil
import pandas as pd
import pyreadr
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine"
# Download latest version
target = DATAPATH
os.makedirs(target, exist_ok=True)
path1 = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
path2 = kagglehub.dataset_download("orvile/leukemia-gene-expression-data-by-golub-et-al")
path3 = kagglehub.dataset_download("gzdekzlkaya/credit-card-fraud-detection-dataset")
for i in os.listdir(path1):
    if i.endswith(".csv"):
        src1 = os.path.join(path1,i)
        dest = os.path.join(target,i)
        shutil.copy(src1, dest)
        print(f"Copied {i} to {target}")

for i in os.listdir(path3):
    if i.endswith(".csv"):
        src3 = os.path.join(path3,i)
        dest = os.path.join(target,i)
        shutil.copy(src3, dest)
        print(f"Copied {i} to {target}")

rda_file = next(Path(path2).glob("*.rda"))
result = pyreadr.read_r(str(rda_file))
train_X = result['golub_train_3051']
train_y = result['golub_train_response'].rename(columns={'golub_train_response': 'label'})
test_X  = result['golub_test_3051']
test_y  = result['golub_test_response'].rename(columns={'golub_test_response': 'label'})
leukemia_df = pd.concat([
    pd.concat([train_X.reset_index(drop=True), train_y.reset_index(drop=True)], axis=1),
    pd.concat([test_X.reset_index(drop=True),  test_y.reset_index(drop=True)],  axis=1),
], ignore_index=True)
dest_csv = target / "leukemia_data.csv"
leukemia_df.to_csv(dest_csv, index=False)
print(f"Converted {rda_file.name} to leukemia_data.csv in {target}")

print("Dataset successfully downloaded.")