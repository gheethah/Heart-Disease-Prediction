import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("heart.csv")
model = joblib.load("RF_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.title("❤️ Heart Disease Analysis & Prediction Dashboard")

# =========================
# TABS (VISUAL + PREDICT)
# =========================
tab1, tab2 = st.tabs(["📊 Data Visualisation", "🧠 Prediction"])

# =========================
# TAB 1 - VISUALISATIONS
# =========================
with tab1:

    st.header("Exploratory Data Analysis")

    # Sidebar filters
    st.sidebar.header("Filters")

    sex_filter = st.sidebar.selectbox(
        "Select Gender",
        ["All"] + list(df["Sex"].unique())
    )

    age_filter = st.sidebar.slider(
        "Age Range",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (30, 70)
    )

    filtered_df = df.copy()

    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["Sex"] == sex_filter]

    filtered_df = filtered_df[
        (filtered_df["Age"] >= age_filter[0]) &
        (filtered_df["Age"] <= age_filter[1])
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heart Disease Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=filtered_df, x="HeartDisease", ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Age Distribution")
        fig2, ax2 = plt.subplots()
        filtered_df["Age"].hist(bins=20, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Cholesterol vs Max Heart Rate")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(
        data=filtered_df,
        x="Cholesterol",
        y="MaxHR",
        hue="HeartDisease",
        ax=ax3
    )
    st.pyplot(fig3)

# =========================
# TAB 2 - PREDICTION
# =========================
with tab2:

    st.header("🧠 Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting BP", 80, 250, 120)
        cholesterol = st.number_input("Cholesterol", 50, 600, 200)
        fasting_bs = st.selectbox("Fasting BS", [0, 1])

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max HR", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # encoding maps
    sex_map = {"M": 1, "F": 0}
    chest_pain_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    resting_ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
    exercise_angina_map = {"Y": 1, "N": 0}
    st_slope_map = {"Up": 0, "Flat": 1, "Down": 2}

    if st.button("Predict"):

        # ✅ EVERYTHING MUST BE INSIDE THIS BLOCK
        sex = sex_map[sex]
        chest_pain = chest_pain_map[chest_pain]
        resting_ecg = resting_ecg_map[resting_ecg]
        exercise_angina = exercise_angina_map[exercise_angina]
        st_slope = st_slope_map[st_slope]

        input_df = pd.DataFrame([[
            age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, resting_ecg, max_hr,
            exercise_angina, oldpeak, st_slope
        ]], columns=[
            "Age","Sex","ChestPainType","RestingBP","Cholesterol",
            "FastingBS","RestingECG","MaxHR",
            "ExerciseAngina","Oldpeak","ST_Slope"
        ])

        num_cols = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        low_risk_prob = prob[0]
        high_risk_prob = prob[1]
        confidence = max(prob)

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.subheader("📊 Model Monitoring Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Prediction",
                value="High Risk" if prediction == 1 else "Low Risk"
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence*100:.2f}%"
            )

        st.subheader("📉 Risk Probability")

        prob_df = pd.DataFrame({
            "Risk Type": ["Low Risk", "High Risk"],
            "Probability": [low_risk_prob, high_risk_prob]
        })

        fig, ax = plt.subplots()
        sns.barplot(data=prob_df, x="Risk Type", y="Probability", ax=ax)
        st.pyplot(fig)