# 🧠 Customer Churn Prediction Dashboard

**Live App:**  
🌐 [Streamlit Dashboard](https://web-based-dashboard-w33ia6kqrvqjiqmxfpwbn9.streamlit.app/)

---

## 📋 Overview

This project is a **web-based churn prediction and retention dashboard** that identifies at-risk customers, visualizes health and risk distributions, and suggests retention strategies.

It provides an **end-to-end workflow** — from data upload, prediction via AWS Lambda, and risk visualization — to retention offer recommendations for the top 5 risky customers.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Model | Logistic Regression |
| Backend | AWS Lambda (Dockerized for deployment) |
| API Gateway | AWS API Gateway (REST endpoint) |
| Hosting | Streamlit Cloud |
| Data Storage (Planned) | AWS S3 + DynamoDB |
| Email (Planned) | AWS SES / Gmail API |

---

## 🧩 Model Performance

| Metric | Value |
|---------|--------|
| Accuracy | 0.978 |
| AUC | 0.995 |
| Model Type | Logistic Regression |
| Deployment Method | Dockerized Lambda (via ECR) due to >250MB package size |

---

## 🖥️ Features

| Feature | Description | Status |
|----------|--------------|---------|
| File Upload (CSV/XLSX) | Upload customer data file for bulk prediction | ✅ Implemented |
| Health Score Distribution | Bar chart for customer health bins | ✅ Implemented |
| Risk Score Distribution | Pie chart (Low/Medium/High) | ✅ Implemented |
| Top 5 Risky Customers | Ranked by combined risk & health | ✅ Implemented |
| Key Churn Drivers | Shows open tickets, usage drop, etc. | ✅ Implemented |
| Suggested Intervention | Automated recommendation (support, discount, etc.) | ✅ Implemented |
| Create Retention Offer | Modal to draft email offer | ✅ Implemented (UI only) |
| Actual Email Sending | Integration with SES/Gmail API | ❌ Not implemented |
| Save Predictions to S3/DynamoDB | Persist prediction results for history | ❌ Missing |
| Refresh Scores | Button to re-fetch latest predictions | ✅ Implemented |
| Authentication (Cognito) | JWT auth for API Gateway | ❌ Optional (not done) |
| Model Accuracy >85% | Verified via backend evaluation | ✅ Achieved |
| Dummy Dataset (1000 train / 500 test) | Custom synthetic dataset | ⚠️ To be uploaded |
| ReadMe Documentation | Requirements, assumptions, limitations | ✅ Done |

---

## 📊 How It Works

### 1️⃣ Upload CSV/XLSX
Users upload customer data.  
Streamlit converts it to JSON and sends it to the Lambda API endpoint.

### 2️⃣ Backend Prediction
AWS Lambda (Dockerized) hosts the Logistic Regression model.  
Returns predicted churn probability, health score, and top churn driver.

### 3️⃣ Frontend Visualization
Streamlit shows:
- Health Score Distribution (Bar chart)
- Risk Score Distribution (Pie chart)
- Top 5 Riskiest Customers with key drivers and interventions
- Allows drafting retention offer emails (mocked, not sent).

### 🔮 Next Planned Enhancements
- Save all predictions to S3 or DynamoDB for future analytics.
- Integrate SES or Gmail API for real email delivery.

---

## 🧰 Local Setup

### Requirements
```
streamlit
pandas
requests
plotly
numpy
openpyxl
```

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧱 AWS Components (Deployed)

| Service | Purpose |
|----------|----------|
| AWS Lambda (Docker) | Model scoring and prediction API |
| API Gateway | REST endpoint for /bulk_predict |
| S3 (planned) | Store uploaded data & prediction results |
| DynamoDB (planned) | Maintain historical prediction logs |
| Streamlit Cloud | Public hosting for frontend dashboard |

---

## ⚠️ Known Limitations

- Email functionality (SES/Gmail API) is mocked — no actual email is sent.
- Predictions are not persisted in S3 or DynamoDB yet.
- Authentication (Cognito) not implemented.
- Model retraining not automated.
- Dummy dataset not yet committed to repository.

---

## 🚀 Next Steps

- [ ] Implement S3 save for uploaded + predicted data  
- [ ] Add SES/Gmail API for live email notifications  
- [ ] Secure API Gateway with Cognito  
- [ ] Automate model retraining pipeline  
- [ ] Improve UI aesthetics and interactivity  

---

## 🧑‍💻 Author

**Aendra Shukla**  
📧 [shuklaaendra123@gmail.com](mailto:shuklaaendra123@gmail.com)  
📧 [aendrashukla@gmail.com](mailto:aendrashukla@gmail.com)
