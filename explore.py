"""
Step 1: Load and explore the Telco Customer Churn dataset.

Download first from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv) in a `data/` folder
next to this script.

Run: python 01_explore.py
"""

import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nChurn distribution:\n", df["Churn"].value_counts(normalize=True))

    # TotalCharges is often read as object due to blank strings - fix it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print("\nRows with bad TotalCharges (now NaN):", df["TotalCharges"].isnull().sum())

    df.to_csv("data/cleaned_telco.csv", index=False)
    print("\nSaved cleaned copy to data/cleaned_telco.csv")

if __name__ == "__main__":
    main()