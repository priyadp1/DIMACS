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
print("Loading dataset...")
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

xgb.fit(X_train, y_train)
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
