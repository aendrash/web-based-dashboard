import json
import pickle
import numpy as np
import pandas as pd

# -----------------------
# Load model + encoders
# -----------------------
with open("model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
ohe = artifact["ohe"]
features = artifact["features"]
num_cols = artifact["num_cols"]
cat_cols = artifact["cat_cols"]

# -----------------------
# Lambda Handler
# -----------------------
def lambda_handler(event, context):
    try:
        # ----------------------------
        # Parse input JSON body
        # ----------------------------
        if "body" not in event:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'body' in event"})
            }

        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, list):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Expected a list of records"})
            }

        df = pd.DataFrame(body)

        # ----------------------------
        # Validate required columns
        # ----------------------------
        required_cols = num_cols + cat_cols
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Missing columns: {missing}"})
            }

        # ----------------------------
        # Preprocess data
        # ----------------------------
        num_values = df[num_cols].to_numpy()
        cat_ohe = ohe.transform(df[cat_cols])
        X = np.hstack([num_values, cat_ohe])

        # ----------------------------
        # Predict churn probabilities
        # ----------------------------
        probs = model.predict_proba(X)[:, 1]
        health_scores = np.round((1 - probs) * 100, 2)

        # ----------------------------
        # Compute top driver
        # ----------------------------
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])

        top_drivers = [
            features[int(np.argmax(importances * row))] for row in X
        ]

        # ----------------------------
        # Prepare response
        # ----------------------------
        df["predicted_churn_prob"] = np.round(probs, 4)
        df["health_score"] = health_scores
        df["top_driver"] = top_drivers

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": df.to_json(orient="records")
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
