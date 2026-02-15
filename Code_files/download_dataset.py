import kagglehub
import os
import shutil

# Download latest version
target = '/Users/prishapriyadashini/Downloads/DIMACS/datasets/Mine/breast_cancer_data.csv'
os.makedirs(target, exist_ok=True)
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
for i in os.listdir(path):
    if i.endswith(".csv"):
        src = os.path.join(path,i)
        dest = os.path.join(target,i)
        shutil.copy(src, dest)
        print(f"Copied {i} to {target}")

print("Dataset successfully downloaded.")