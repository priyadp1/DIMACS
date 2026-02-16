import time
import pandas as pd
from resplit.model.treefarms import TREEFARMS
from resplit import RESPLIT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
path = DATAPATH
print(f"Loading dataset from: {path}")
df = pd.read_csv(path)
print("Mapping diagnosis to binary...")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("Preparing features and labels...")
df = df.dropna(axis=1, how="all")
X = df.drop(columns=["id", "diagnosis"])
Y = df["diagnosis"]
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


