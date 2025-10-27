# train_model.py
import pandas as pd
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# read
print("reading csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# features
cat_cols = ['segment', 'region']
num_cols = [
    'avg_daily_usage', 'usage_drop_pct', 'open_tickets', 'failed_payments',
    'days_since_last_login', 'tenure_months', 'mrr',
    'recent_support_escalations', 'payment_streak'
]
print("aboce ohe")
# one-hot encode categories
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(train[cat_cols])
train_ohe = ohe.transform(train[cat_cols])
test_ohe = ohe.transform(test[cat_cols])
ohe_cols = ohe.get_feature_names_out(cat_cols).tolist()
print("training")
# combine features
X_train = np.hstack([train[num_cols].values, train_ohe])
X_test = np.hstack([test[num_cols].values, test_ohe])
features = num_cols + ohe_cols

y_train = train['churn_label']
y_test = test['churn_label']

# --------------------------
# LIGHTWEIGHT MODEL
# --------------------------
model = LogisticRegression(
    max_iter=500,           # ensure convergence
    solver='lbfgs',         # efficient for medium-sized dense data
    penalty='l2',           # regularized logistic regression
    C=1.0,                  # inverse of regularization strength
    random_state=42
)
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)
print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")

# save artifact
with open("project/model.pkl", "wb") as f:
    pickle.dump(
        {
            'model': model,
            'ohe': ohe,
            'features': features,
            'num_cols': num_cols,
            'cat_cols': cat_cols
        },
        f
    )

print("Saved model.pkl")

# save feature importances (using coefficients)
importances = list(zip(features, model.coef_[0]))
importances = [(f, float(abs(v))) for f, v in importances]  # absolute importance
importances = sorted(importances, key=lambda x: x[1], reverse=True)

with open("project/feature_importances.json", "w") as f:
    json.dump(importances, f, indent=2)

print("feature importances saved")
