import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "creditcard_fraud_detection.csv"

df = pd.read_csv(DATAPATH)
print("Original class distribution:\n", df['Class'].value_counts())
print(f"\nFraud rate: {df['Class'].mean():.4%}")

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Time and Amount (V1-V28 are already PCA-normalized)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

print("\nBefore SMOTE, class distribution in training set:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE, class distribution in training set:\n", pd.Series(y_train_resampled).value_counts())

# Save resampled training set
df_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
df_resampled['Class'] = y_train_resampled
OUTPATH = BASEDIR / "datasets" / "Mine" / "creditcard_fraud_detection_smote.csv"
df_resampled.to_csv(OUTPATH, index=False)
print(f"\nSaved SMOTE-resampled training data to {OUTPATH}")

# Save scaled test set separately for consistent evaluation
df_test = pd.DataFrame(X_test, columns=X_test.columns)
df_test['Class'] = y_test.values
TESTPATH = BASEDIR / "datasets" / "Mine" / "creditcard_fraud_detection_test.csv"
df_test.to_csv(TESTPATH, index=False)
print(f"Saved scaled test data to {TESTPATH}")
