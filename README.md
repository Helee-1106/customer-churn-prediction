# Customer Churn Prediction

Predicts telecom customer churn using classification models, comparing
Logistic Regression against Random Forest.

## Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
~7,000 customers, 20 features (contract type, tenure, charges, services used, etc.)

## How it works
1. **`01_explore.py`** — loads data, checks missing values, fixes `TotalCharges`
   column type, reports class balance
2. **`02_train_models.py`** — encodes features, trains Logistic Regression and
   Random Forest, evaluates both, saves confusion matrix + feature importance charts

## Results
*(Run `02_train_models.py` and paste your actual numbers below — no placeholders)*

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | [FILL IN] | [FILL IN] | [FILL IN] |
| Random Forest | [FILL IN] | [FILL IN] | [FILL IN] |

**Top churn drivers:** [FILL IN 2-3 features from feature_importance.png, e.g.
"contract type, tenure, and monthly charges were the strongest predictors of churn"]

![Confusion Matrix](confusion_matrix.png)
![Feature Importance](feature_importance.png)

## Skills demonstrated
Data cleaning, feature encoding, train/test splitting, classification modeling,
model comparison, evaluation metrics (F1, ROC-AUC), feature importance analysis

## How to run
```bash
pip install pandas scikit-learn matplotlib seaborn
python 01_explore.py      # place Telco CSV in data/ first
python 02_train_models.py
```
