"""
Streamlit App - Wellness Tourism Package Prediction
-----------------------------------------------------
Loads the trained pipeline (preprocessing + best classifier) and lets a
salesperson enter a customer's profile to predict whether that customer is
likely to buy the new Wellness Tourism Package.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HF_USERNAME = "abhayfps1"          # <-- TODO: replace with your HF username
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction-model"
LOCAL_MODEL_PATH = "best_tourism_model_v1.joblib"  # bundled alongside app.py


@st.cache_resource
def load_model():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename="best_tourism_model_v1.joblib",
                token=hf_token,
            )
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load model from Hugging Face Hub ({e}); using local copy.")
    return joblib.load(LOCAL_MODEL_PATH)


st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘", layout="centered")
st.title("🧘 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter a customer's profile below to predict whether they are likely "
    "to purchase the new Wellness Tourism Package."
)

model = load_model()

with st.form("customer_form"):
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthly_income = st.number_input("Monthly Income", min_value=0, value=20000, step=500)
    with col2:
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
        num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=25, value=3)
        preferred_star = st.selectbox("Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
        passport = st.selectbox("Holds Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

    st.subheader("Sales Interaction Details")
    col3, col4 = st.columns(2)
    with col3:
        product_pitched = st.selectbox(
            "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
        )
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
    with col4:
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Likely to purchase the Wellness Package (confidence: {probability:.1%})")
    else:
        st.info(f"❌ Unlikely to purchase the Wellness Package (confidence: {1 - probability:.1%})")
