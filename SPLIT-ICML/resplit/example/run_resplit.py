import time
import pandas as pd
from resplit.model.treefarms import TREEFARMS
from resplit import RESPLIT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
from pathlib import Path
import os
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
path = DATAPATH
results_dir = BASEDIR/"model_results"
os.makedirs(results_dir , exist_ok=True)
import json as _json
_cfg_file = BASEDIR / "_run_config.json"
if _cfg_file.exists():
    with open(_cfg_file) as _f:
        _cfg = _json.load(_f)
    DATAPATH    = Path(_cfg['dataset_path'])
    path        = DATAPATH
    results_dir = Path(_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    _target_col = _cfg['target_column']
    _drop_cols  = _cfg['drop_columns']
    _label_map  = _cfg.get('label_map')
else:
    _target_col = 'diagnosis'
    _drop_cols  = ['id', 'diagnosis']
    _label_map  = {'M': 1, 'B': 0}
print(f"Loading dataset from: {path}")
df = pd.read_csv(path)
print("Mapping diagnosis to binary...")
if _label_map:
    df[_target_col] = df[_target_col].map(_label_map)

print("Preparing features and labels...")
df = df.dropna(axis=1, how="all")
X = df.drop(columns=_drop_cols)
Y = df[_target_col]
print("X shape:" , X.shape)
print("Y dist:\n" , Y.value_counts())
 
X_train, X_test, y_train, y_test = train_test_split(
    X,Y,test_size=0.2, random_state=42, stratify=Y
)

print("\n Running RESPLIT...")
config = {
        "regularization": 0.01,
        "rashomon_bound_multiplier": 0.01,
        "depth_budget": 3,
        "cart_lookahead_depth": 1,
        "verbose": True
    }
model = RESPLIT(config, fill_tree='treefarms')
start = time.perf_counter()
model.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Done training RESPLIT")
print("Making predicitions...")
y_pred = model.predict(X_test, idx=0)
with open(results_dir / "resplit_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nRESPLIT completed in {duration:.2f} seconds with {len(model)} trees")
print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))
print(f" RESPLIT completed in {duration:.2f} seconds with {len(model)} trees")


print("\n Running TREEFARMS...")
config = {
        "regularization": 0.01,
        "rashomon_bound_multiplier": 0.01,
        "depth_budget": 3,
        "verbose": True
    }
model = TREEFARMS(config)
start = time.perf_counter()
model.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Done training Treefarms")
print("Making predicitions...")
tree = model[0]
y_pred = tree.predict(X_test)
print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))
print(f" TREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
with open(results_dir / "treefarms_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nTREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")


