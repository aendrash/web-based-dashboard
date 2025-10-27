# generate_dataset.py
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

N = 1500
data = []
segments = ['enterprise','mid-market','small-business']
regions = ['north','south','east','west']
for i in range(N):
    cid = f"CUST{i:05d}"
    segment = random.choices(segments, weights=[0.2,0.5,0.3])[0]
    region = random.choice(regions)
    # features
    avg_daily_usage = max(0, np.random.normal(100 if segment=='enterprise' else 50 if segment=='mid-market' else 20, 30))
    usage_drop_pct = max(0, np.random.normal(0 if avg_daily_usage>80 else 20, 10))
    open_tickets = int(np.random.poisson(0.2 if segment=='enterprise' else 0.8 if segment=='mid-market' else 1.5))
    failed_payments = int(np.random.binomial(2, 0.02 if segment=='enterprise' else 0.08 if segment=='mid-market' else 0.12))
    days_since_last_login = int(max(0, np.random.exponential(10 if avg_daily_usage>60 else 30)))
    tenure_months = int(np.random.randint(1,60))
    mrr = avg_daily_usage * (100 if segment=='enterprise' else 10 if segment=='mid-market' else 2)
    # engineered features
    recent_support_escalations = 1 if open_tickets>2 else 0
    payment_streak = max(0, 12 - failed_payments*3 - int(days_since_last_login/30))
    # churn label logic (intentionally separable)
    risk_score = 0.4*(usage_drop_pct/50) + 0.3*(open_tickets/5) + 0.2*(failed_payments/2) + 0.1*(min(1, days_since_last_login/60))
    churn = 1 if risk_score > 0.35 or (failed_payments>0 and usage_drop_pct>10) else 0
    health_score = int(max(0, min(100, 100 - (risk_score*100))))
    data.append({
        'customer_id': cid,
        'segment': segment,
        'region': region,
        'avg_daily_usage': round(avg_daily_usage,2),
        'usage_drop_pct': round(usage_drop_pct,2),
        'open_tickets': open_tickets,
        'failed_payments': failed_payments,
        'days_since_last_login': days_since_last_login,
        'tenure_months': tenure_months,
        'mrr': round(mrr,2),
        'recent_support_escalations': recent_support_escalations,
        'payment_streak': payment_streak,
        'churn_label': churn,
        'health_score': health_score
    })

df = pd.DataFrame(data)
train, test = train_test_split(df, test_size=500, random_state=42, stratify=df['churn_label'])
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
print("Generated train.csv ({} rows) and test.csv ({} rows)".format(len(train), len(test)))
