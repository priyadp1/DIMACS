import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR/"datasets/Mine/breast_cancer_data.csv"

# Parameters
GBDT_N_EST = 40
GBDT_MAX_DEPTH = 1
REGULARIZATION = 0.001
SIMILAR_SUPPORT = False
DEPTH_BUDGET = 6
TIME_LIMIT = 60
VERBOSE = True

# Read the dataset
df = pd.read_csv(DATAPATH)
df = df.dropna(axis=1, how="all")
print("Mapping diagnosis to binary...")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print("Preparing features and labels...")
X = df.drop(columns=["id", "diagnosis"])
Y = df["diagnosis"]
print("X shape:" , X.shape)
print("Y dist:\n" , Y.value_counts())
h = X.columns

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
print("X train shape:{}, X test shape:{}".format(X_train.shape, X_test.shape))

# Step 1: Guess Thresholds
enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
enc.set_output(transform="pandas")
X_train_guessed = enc.fit_transform(X_train, y_train)
X_test_guessed = enc.transform(X_test)
print(f"After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}")
print(f"train set column names == test set column names: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}")

# Step 2: Guess Lower Bounds
enc = GradientBoostingClassifier(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
enc.fit(X_train_guessed, y_train)
warm_labels = enc.predict(X_train_guessed)

# Step 3: Train the GOSDT classifier
clf = GOSDTClassifier(regularization=REGULARIZATION, similar_support=SIMILAR_SUPPORT, time_limit=TIME_LIMIT, depth_budget=DEPTH_BUDGET, verbose=VERBOSE) 
clf.fit(X_train_guessed, y_train, y_ref=warm_labels)
print("Done training")
print("Making predicitions...")
y_pred = clf.predict(X_test_guessed)

# Step 4: Evaluate the model
print("Evaluating the model, extracting tree and scores", flush=True)


print(f"Model training time: {clf.result_.time}")
print(f"Training accuracy: {clf.score(X_train_guessed, y_train)}")
print(f"Test accuracy: {clf.score(X_test_guessed, y_test)}")
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))