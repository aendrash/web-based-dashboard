import streamlit as st
import pandas as pd
import requests
import json
import io
import numpy as np
import plotly.express as px

API_BULK_URL = "https://8cdh766frh.execute-api.ap-south-1.amazonaws.com/v1/bulk_predict"

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("📊 Bulk Customer Churn Prediction Dashboard")

# --- Upload Section ---
st.subheader("📂 Upload CSV or Excel file")
uploaded_file = st.file_uploader("Upload your customer data file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Loaded file: {uploaded_file.name}")
        # st.write("### Preview of Uploaded Data")
        # st.dataframe(df.head())

        # Convert to JSON list
        records = df.to_dict(orient="records")
        # st.write("🧾 **JSON Payload Preview (first 2 records)**")
        # st.json(records[:2])

        if st.button("🚀 Run Bulk Prediction"):
            try:
                with st.spinner("Running predictions..."):
                    response = requests.post(API_BULK_URL, json=records)
                    response.raise_for_status()
                    predictions = response.json()

                # ✅ Validation
                if isinstance(predictions, list):
                    result_df = pd.DataFrame(predictions)

                    # --- Upload Summary ---
                    st.success(f"✅ Upload Complete! {len(result_df)} predictions generated.")
                    st.info(
                        f"Processed: {len(df)} rows | Valid: {len(result_df)} | Invalid: {len(df) - len(result_df)}")

                    # --- Health Score Chart ---
                    st.subheader("📈 Health Score Distribution")
                    bins = [0, 20, 40, 60, 80, 100]
                    result_df["health_bin"] = pd.cut(result_df["health_score"], bins=bins)
                    health_chart = result_df["health_bin"].value_counts().sort_index()
                    fig_health = px.bar(
                        x=health_chart.index.astype(str),
                        y=health_chart.values,
                        labels={"x": "Health Score Range", "y": "Customer Count"},
                        title="Health Score Distribution"
                    )
                    st.plotly_chart(fig_health, use_container_width=True)

                    # --- Risk Calculation ---
                    result_df["risk_score"] = (result_df["predicted_churn_prob"] * 100).round(2)
                    result_df["final_risk"] = ((100 - result_df["health_score"]) * 0.5 +
                                               result_df["risk_score"] * 0.5).round(2)

                    # --- Risk Score Distribution ---
                    st.subheader("🔥 Risk Score Distribution")
                    risk_bins = [0, 40, 70, 100]
                    risk_labels = ["Low (<40)", "Medium (40-70)", "High (>70)"]
                    result_df["risk_level"] = pd.cut(result_df["final_risk"], bins=risk_bins, labels=risk_labels,
                                                     include_lowest=True)
                    risk_chart = result_df["risk_level"].value_counts().reindex(risk_labels)
                    fig_risk = px.pie(
                        names=risk_chart.index,
                        values=risk_chart.values,
                        title="Risk Level Distribution",
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                    # --- Top 5 Risky Customers ---
                    st.subheader("⚠️ Top 5 Riskiest Customers")
                    top5 = result_df.sort_values("final_risk", ascending=False).head(5)
                    top5["intervention"] = top5["top_driver"].apply(lambda d:
                                                                    "Offer dedicated support" if "ticket" in d else
                                                                    "Provide discount or grace period" if "payment" in d else
                                                                    "Re-engagement email" if "login" in d else
                                                                    "Feature demo / usage training"
                                                                    )

                    st.dataframe(top5[["customer_id", "health_score", "risk_score", "final_risk",
                                       "top_driver", "intervention"]])

                    # --- Email Offer Modal ---
                    st.write("### ✉️ Create Retention Offer")
                    selected_customer = st.selectbox("Select Customer", top5["customer_id"].tolist())

                    if selected_customer:
                        cust = top5[top5["customer_id"] == selected_customer].iloc[0]
                        st.info(f"Creating offer for **{cust['customer_id']}** | Key issue: {cust['top_driver']}")
                        suggested = cust["intervention"]

                        subject = st.text_input("Subject", f"We're Here to Help You, {cust['customer_id']}")
                        message = st.text_area("Email Content",
                                               f"""Hi {cust['customer_id']} Team,

                            We noticed an issue: {cust['top_driver']}.
                            Suggested Action: {suggested}.

                            We’d love to help you get more value from our platform.

                            Best,
                            Customer Success Team""")

                        if st.button("📤 Send Email"):
                            st.success(f"✅ Retention offer sent to {cust['customer_id']}!")
                            st.toast(f"Email sent to {cust['customer_id']} successfully.")

                    # --- Refresh Button ---
                    if st.button("🔄 Refresh Scores"):
                        st.experimental_rerun()

                else:
                    st.error(f"Unexpected response format: {predictions}")

            except Exception as e:
                st.error(f"API Error: {e}")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
