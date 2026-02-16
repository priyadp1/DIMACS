from split import SPLIT
import pandas as pd
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"

print("Loading dataset...")
lookahead_depth = 2
depth_buget = 5
regularization = 0.01

dataset = pd.read_csv(DATAPATH) 
dataset = dataset.dropna(axis=1, how="all")

print("Mapping diagnosis to binary...")
dataset["diagnosis"] = dataset["diagnosis"].map({"M": 1, "B": 0})
print("Preparing features and labels...")

X = dataset.drop(columns=["id", "diagnosis"])
Y = dataset["diagnosis"]
print("X shape:" , X.shape)
print("Y dist:\n" , Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,Y,test_size=0.2, random_state=42, stratify=Y
)

# y should correspond to a binary class label. 
print("Initializing SPLIT...")
model = SPLIT(lookahead_depth_budget=lookahead_depth, reg=regularization, full_depth_budget=depth_buget, verbose=True, binarize=True,time_limit=100)
# set binarize = True if dataset is not binarized.
print("Starting training...")
model.fit(X_train,y_train)

print("Done training")
print("Making predicitions...")
y_pred = model.predict(X_test)

print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))
print("Tree structure:")
print(model.tree)