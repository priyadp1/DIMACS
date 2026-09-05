import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "diabetic_data.csv"
df = pd.read_csv(DATAPATH)
print("Original class distribution:\n", df['readmitted'].value_counts())
X = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'readmitted'])
y = df['readmitted'].map({">30": 1, "<30": 1, "NO": 0})
# Encode categorical columns so SMOTE can process them
cat_cols = X.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nBefore SMOTE Tomek, class distribution in training set:\n", y_train.value_counts())
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
print("\nAfter SMOTE Tomek, class distribution in training set:\n", pd.Series(y_train_resampled).value_counts())
# Combine resampled training set into a DataFrame and save
df_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
df_resampled['readmitted'] = y_train_resampled
OUTPATH = BASEDIR / "datasets" / "Mine" / "diabetes_smote_tomek.csv"
df_resampled.to_csv(OUTPATH, index=False)
print(f"\nSaved SMOTE-Tomek-resampled training data to {OUTPATH}")