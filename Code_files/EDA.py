import pandas as pd

df = pd.read_csv("/Users/prishapriyadashini/Downloads/DIMACS/datasets/Mine/breast_cancer_data.csv")
print("\nInformation: ")
df.info()
print("\n Statistics: ")
print(df.describe())
print("\n Number of Null Values: ")
print(df.isna().sum())
print("\nShape: ")
print(df.shape)
print("\nValue Counts")
for i in df.columns:
    print(f"{i}: ")
    print(df[i].value_counts())