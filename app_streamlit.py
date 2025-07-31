import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pandas as pd

# Load model and features
model = joblib.load("loan_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Configure page
st.set_page_config(page_title="Loan Eligibility App", layout="centered")

# Session state for navigation
def go_to_result(prediction, inputs):
    st.session_state.prediction_result = prediction
    st.session_state.input_summary = inputs  # store inputs for later display
    st.session_state.page = "result"
    st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "input_summary" not in st.session_state:
    st.session_state.input_summary = None

# Navigation handlers
def go_to_form():
    st.session_state.page = "form"
    st.rerun()

def go_to_home():
    st.session_state.page = "home"
    st.rerun()

# Page 1: Landing
if st.session_state.page == "home":
    st.markdown("""
    <style>
    body {
        margin-top: 0 !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
    }
    .simple-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin: 1rem 0 0.5rem 0;
        color: white;
    }
    .simple-subtitle {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1rem;
        color: #cccccc;
    }
    .center-btn {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

    st.markdown('<div class="simple-title">Check Your Loan Approval</div>', unsafe_allow_html=True)
    st.markdown('<div class="simple-subtitle">Find out if you are eligible for a personal loan in seconds</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ CHECK ELIGIBILITY"):
                go_to_form()

    try:
        card_img = Image.open("images/loan.png")  
        st.image(card_img, use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Image not found. Check the filename or path.")

# Page 2: Form
elif st.session_state.page == "form":
    if st.button("⬅ Go Back"):
        go_to_home()

    st.title("📋 Loan Application Form")
    st.markdown("Fill in your details below to check loan approval status:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_amount_term = st.number_input("Loan Term (in days)", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])

    if st.button("🔍 Predict"):
        total_income = applicant_income + coapplicant_income
        loan_income_ratio = loan_amount / total_income if total_income != 0 else 0
        dependents_num = 3 if dependents == "3+" else int(dependents)
        property_area_semiurban = 1 if property_area == "Semiurban" else 0
        property_area_urban = 1 if property_area == "Urban" else 0

        input_dict = {
            "Gender": 1 if gender == "Male" else 0,
            "Married": 1 if married == "Yes" else 0,
            "Dependents": dependents_num,
            "Education": 1 if education == "Graduate" else 0,
            "Self_Employed": 1 if self_employed == "Yes" else 0,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area_Semiurban": property_area_semiurban,
            "Property_Area_Urban": property_area_urban,
            "Total_Income": total_income,
            "Loan_Income_Ratio": loan_income_ratio
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        go_to_result(prediction, input_dict)

# ---------- PAGE 3: Result ----------
elif st.session_state.page == "result":
    st.title("📊 Loan Eligibility Result")

    result = st.session_state.prediction_result

    if result == 1:
        st.success("🎉 Congratulations! You are likely to be approved for a loan.")
        try:
            st.image("images/success.png", use_container_width=True)
        except:
            st.info("✔️ Loan approved visuals here.")
    else:
        st.error("❌ Sorry! You are unlikely to be approved at this time.")
        try:
            st.image("images/failure.png", use_container_width=True)
        except:
            st.info("📄 Try different values or improve credit factors.")

    # Optional input summary
    if st.session_state.input_summary:
        with st.expander("📄 See your submitted data"):
            st.json(st.session_state.input_summary)

    st.markdown("---")

    st.markdown("""
        <style>
        .custom-button {
            background-color: #262730;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            color: white;
            font-size: 1rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        .button-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create button row layout
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔁 Try Again", key="try_again_btn"):
            st.session_state.page = "form"
            st.rerun()

    with col2:
        # Right-aligned home button with same style
        if st.button("🏠 Home", key="home_btn"):
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)



