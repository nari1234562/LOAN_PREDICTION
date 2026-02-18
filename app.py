import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline model
with open("model1.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Loan Default Prediction System")

st.subheader("Enter Applicant Details")

# ---- NUMERICAL INPUTS ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
employment_experience_years = st.number_input("Employment Experience (Years)", min_value=0.0, value=2.0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
loan_to_income_ratio = st.number_input("Loan to Income Ratio", min_value=0.0, value=0.2)
credit_history_length_years = st.number_input("Credit History Length (Years)", min_value=0.0, value=3.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

# ---- CATEGORICAL INPUTS ----
gender = st.selectbox("Gender", ["male", "female"])
education_level = st.selectbox("Education Level", 
                               ["High School", "Associate", "Bachelor", "Master", "Doctorate"])

home_ownership_status = st.selectbox("Home Ownership Status",
                                      ["RENT", "OWN", "MORTGAGE", "OTHER"])

loan_purpose = st.selectbox("Loan Purpose",
                            ["PERSONAL", "EDUCATION", "MEDICAL", 
                             "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

prior_default_flag = st.selectbox("Previous Loan Default",
                                  ["YES", "NO"])

# ---- CREATE INPUT DATAFRAME ----
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "education_level": [education_level],
    "annual_income": [annual_income],
    "employment_experience_years": [employment_experience_years],
    "home_ownership_status": [home_ownership_status],
    "loan_amount": [loan_amount],
    "loan_purpose": [loan_purpose],
    "interest_rate": [interest_rate],
    "loan_to_income_ratio": [loan_to_income_ratio],
    "credit_history_length_years": [credit_history_length_years],
    "credit_score": [credit_score],
    "prior_default_flag": [prior_default_flag]
})

# ---- PREDICTION ----
if st.button("Predict Default Risk"):
    
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Default Probability: {probability:.2f}")

    # Custom threshold = 0.6
    if probability > 0.6:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk - Loan Can Be Approved")

