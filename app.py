import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========= 1. LOAD MODEL, SCALER, AND DATA =========
@st.cache_resource
def load_artifacts():
    # load trained model and scaler
    scaler = joblib.load("model_scaler.joblib")
    model = joblib.load("model_random_forest.joblib")

    # load dataset to get feature names and simple stats
    df = pd.read_csv("data.csv")

    # drop columns that were not used during training
    for col in ["id", "Unnamed: 32", "diagnosis"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    feature_names = df.columns.tolist()
    stats = df.describe()  # for min/max/mean
    return scaler, model, feature_names, stats


scaler, model, FEATURE_NAMES, STATS = load_artifacts()

# ========= 2. STREAMLIT PAGE CONFIG =========
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Breast Cancer Detection App")
st.write(
    """
This app uses a **Machine Learning model (Random Forest)** trained on the 
*Breast Cancer Wisconsin (Diagnostic)* dataset to predict whether a tumor 
is likely **Benign (non-cancerous)** or **Malignant (cancerous)**.

> ⚠️ **Disclaimer:** This tool is for **learning and demonstration only**  
> and **not** for real medical diagnosis or treatment decisions.
"""
)

st.markdown("---")

# ========= 3. INPUT FORM =========
st.subheader("🔢 Enter Tumor Measurements")

st.write(
    "You can adjust the values below. Defaults are close to the **average** values from the dataset."
)

with st.form("input_form"):
    user_values = []

    for feature in FEATURE_NAMES:
        # basic stats for the feature
        mean_val = float(STATS.loc["mean", feature])
        min_val = float(STATS.loc["min", feature])
        max_val = float(STATS.loc["max", feature])

        # make the input range a bit wider than min/max
        range_margin = (max_val - min_val) * 0.1
        display_min = min_val - range_margin
        display_max = max_val + range_margin

        # Use number_input (can also use sliders but 30 sliders is heavy)
        val = st.number_input(
            label=feature,
            min_value=float(display_min),
            max_value=float(display_max),
            value=float(mean_val),
            step=float((max_val - min_val) / 100 if max_val != min_val else 0.01),
            format="%.4f",
        )
        user_values.append(val)

    submitted = st.form_submit_button("🔍 Predict")


# ========= 4. PREDICTION =========
if submitted:
    # create DataFrame with one row
    input_df = pd.DataFrame([user_values], columns=FEATURE_NAMES)

    # scale features
    input_scaled = scaler.transform(input_df)

    # model prediction
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # probability of class 1 (malignant)

    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    if pred == 1:
        st.error(
            f"**Model Prediction: Malignant (Cancerous Tumor)**\n\n"
            f"Estimated probability of malignancy: **{proba * 100:.2f}%**"
        )
    else:
        st.success(
            f"**Model Prediction: Benign (Non-cancerous Tumor)**\n\n"
            f"Estimated probability of malignancy: **{proba * 100:.2f}%**"
        )

    st.caption(
        "Higher probability for malignant means higher risk according to the model. "
        "Always consult a medical professional for real-world decisions."
    )

# ========= 5. SIDEBAR INFO =========
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.write(
        """
- Algorithm: **Random Forest Classifier**
- Trained on: **Breast Cancer Wisconsin (Diagnostic)** dataset  
- Features: **30 tumor-related measurements**
- Target: Benign (0) vs Malignant (1)
        """
    )

    st.markdown("---")
    st.write("👩‍💻 *Built with Python, Scikit-learn & Streamlit*")
