"""
Step 2: Train and compare Logistic Regression vs Random Forest
on the Telco Churn dataset. Prints real metrics you'll use in your README.

Run: python 02_train_models.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

def load_and_prep():
    df = pd.read_csv("data/cleaned_telco.csv")

    # Drop customerID (not predictive), drop rows with missing TotalCharges
    df = df.drop(columns=["customerID"])
    df = df.dropna(subset=["TotalCharges"])

    # Target
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    # Encode categorical columns
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    return X, y

def main():
    X, y = load_and_prep()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "f1": f1_score(y_test, lr_preds),
        "roc_auc": roc_auc_score(y_test, lr_probs),
    }

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, rf_preds),
        "f1": f1_score(y_test, rf_preds),
        "roc_auc": roc_auc_score(y_test, rf_probs),
    }

    print("=== Model Comparison (use these real numbers in your README) ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")

    print("\n=== Random Forest Classification Report ===")
    print(classification_report(y_test, rf_preds))

    # --- Confusion matrix plot (Random Forest) ---
    cm = confusion_matrix(y_test, rf_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Random Forest — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # --- Feature importance plot ---
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", color="steelblue")
    plt.title("Top 10 Feature Importances — Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    print("\nSaved: confusion_matrix.png, feature_importance.png")
    print("\nCopy the metrics above into your README — do not use placeholder numbers.")

if __name__ == "__main__":
    main()