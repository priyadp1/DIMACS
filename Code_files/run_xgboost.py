import os
import time
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
results_dir = BASEDIR / "model_results"
os.makedirs(results_dir, exist_ok=True)
import json as _json
_cfg_file = BASEDIR / "_run_config.json"
if _cfg_file.exists():
    with open(_cfg_file) as _f:
        _cfg = _json.load(_f)
    DATAPATH    = Path(_cfg['dataset_path'])
    results_dir = Path(_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    _target_col = _cfg['target_column']
    _drop_cols  = _cfg['drop_columns']
    _label_map  = _cfg.get('label_map')
else:
    _target_col = 'diagnosis'
    _drop_cols  = ['id', 'diagnosis']
    _label_map  = {'M': 1, 'B': 0}
print("Loading dataset...")
dataset = pd.read_csv(DATAPATH) 
dataset = dataset.dropna(axis=1, how="all")

print("Mapping diagnosis to binary...")
if _label_map:
    dataset[_target_col] = dataset[_target_col].map(_label_map)
print("Preparing features and labels...")

X = dataset.drop(columns=_drop_cols)
Y = dataset[_target_col]
print("X shape:" , X.shape)
print("Y dist:\n" , Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,Y,test_size=0.2, random_state=42, stratify=Y
)

# XGBoost disallows [ ] < in feature names — replace with safe characters
X_train.columns = X_train.columns.str.replace('[', '{', regex=False).str.replace(']', '}', regex=False).str.replace('<', 'lt', regex=False)
X_test.columns  = X_test.columns.str.replace('[', '{', regex=False).str.replace(']', '}', regex=False).str.replace('<', 'lt', regex=False)
print("\n Training XGBoost")

xgb = XGBClassifier(
    max_depth=3,
    n_estimators=25,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    reg_alpha=0.0,
    eval_metric="logloss",
    random_state=42
)

start = time.perf_counter()
xgb.fit(X_train, y_train)
duration = time.perf_counter() - start
y_pred = xgb.predict(X_test)
print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))

print("\n Feature Importance: ")
importances = xgb.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance" : importances
}).sort_values(by="Importance" , ascending=False)
print("\n Importance Df: ")
print(importance_df.head(3))
try:
    _trees_df = xgb.get_booster().trees_to_dataframe()
    _n_leaves = int((_trees_df['Feature'] == 'Leaf').sum())
    _n_nodes = int(len(_trees_df))
    _n_trees = int(_trees_df['Tree'].nunique())
    _tree_size = {
        "n_trees": _n_trees,
        "total_leaves": _n_leaves,
        "total_nodes": _n_nodes,
        "avg_leaves_per_tree": round(_n_leaves / _n_trees, 2),
    }
except Exception as _e:
    _tree_size = {"error": str(_e)}
with open(results_dir / "xgboost_tree_size.json", "w") as f:
    import json as _json_sz; _json_sz.dump(_tree_size, f)

with open(results_dir / "xgboost_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nXGBoost completed in {duration:.2f} seconds")
    f.write(f"\nTop 3 Features:\n{importance_df.head(3).to_string(index=False)}")
    if "error" not in _tree_size:
        f.write(f"\nTree Size: {_tree_size['n_trees']} trees, {_tree_size['total_leaves']} total leaves, {_tree_size['avg_leaves_per_tree']:.1f} avg leaves/tree")
    else:
        f.write(f"\nTree Size: Error - {_tree_size['error']}")
