import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
for col in df.select_dtypes(exclude='number').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X, y = df.drop('Churn', axis=1), df['Churn']
print("Rows:", len(df), "| Churn rate: %.1f%%" % (100 * y.mean()))

# stratify keeps the churn ratio the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Baseline: logistic regression (scaling is inside the pipeline, so no leakage)
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)
print("LogReg test ROC-AUC: %.4f" % roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]))

# 2. XGBoost, with class imbalance handled
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                    eval_metric='logloss', random_state=42, scale_pos_weight=spw)
xgb.fit(X_train, y_train)
prob = xgb.predict_proba(X_test)[:, 1]
pred = xgb.predict(X_test)
print("XGB test ROC-AUC: %.4f" % roc_auc_score(y_test, prob))
print("XGB precision: %.2f | recall: %.2f | F1: %.2f" % (
    precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)))

# 3. 5-fold stratified cross-validation on the TRAINING data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in [("LogReg", logreg), ("XGBoost", xgb)]:
    s = cross_val_score(m, X_train, y_train, cv=cv, scoring='roc_auc')
    print("%s 5-fold CV ROC-AUC: %.4f +/- %.4f" % (name, s.mean(), s.std()))
    raw = pd.read_csv(url)
raw['Churn'] = raw['Churn'].map({'Yes': 1, 'No': 0})
print(raw.groupby('Contract')['Churn'].mean().mul(100).round(1))
print(raw.groupby(raw['tenure'] < 12)['Churn'].mean().mul(100).round(1))