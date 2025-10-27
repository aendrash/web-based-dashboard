import pickle, pandas as pd, numpy as np, json

# Load model artifact
with open("project/model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
ohe = artifact["ohe"]
features = artifact["features"]
num_cols = artifact["num_cols"]
cat_cols = artifact["cat_cols"]

# --- helper to score one record ---
def score_customer(record: dict):
    """record: dict with all numeric + categorical fields"""
    num_values = np.array([[record[col] for col in num_cols]])
    cat_values = pd.DataFrame([[record[c] for c in cat_cols]], columns=cat_cols)
    cat_ohe = ohe.transform(cat_values)
    X = np.hstack([num_values, cat_ohe])

    prob = float(model.predict_proba(X)[:, 1])
    health_score = round((1 - prob) * 100, 2)

    # --- simple driver extraction heuristic ---
    importances = model.feature_importances_
    contrib = importances * X[0]
    top_idx = int(np.argmax(contrib))
    top_driver = features[top_idx]

    return {
        "predicted_churn_prob": round(prob, 4),
        "health_score": health_score,
        "top_driver": top_driver
    }


# --- Example test with one record from your test.csv ---
df = pd.read_csv("test.csv").iloc[0]
record = df.to_dict()
result = score_customer(record)
print(json.dumps(result, indent=2))
