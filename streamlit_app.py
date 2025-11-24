import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ================================
# CARGAR ARTEFACTOS
# ================================
ART_DIR = "artefactos"

PIPE = joblib.load(os.path.join(ART_DIR, "pipeline_RG.joblib"))
INPUT_SCHEMA = json.load(open(os.path.join(ART_DIR, "input_schema.json"), "r"))
POLICY = json.load(open(os.path.join(ART_DIR, "decision_policy.json"), "r"))

schema_cols = INPUT_SCHEMA["columns"]
lower = POLICY.get("lower")
upper = POLICY.get("upper")

st.title("💼 Predicción de Work-Life Balance (WLB)")
st.subheader("Completa los datos del usuario")


# ================================
# VARIABLES DEL DATASET REAL
# ================================
gender = st.selectbox("Gender:", ["Male", "Female"])
age = st.selectbox("Age group:", ["Less than 20", "21 to 35", "36 to 50", "51 or more"])
stress = st.selectbox("Daily Stress (0–5):", [0,1,2,3,4,5])

sleep = st.slider("Sleep Hours", 0, 12, 7)
steps = st.slider("Daily Steps", 0, 20000, 7000)
fruits = st.slider("Fruits & Veggies Intake", 0, 10, 3)
flow = st.slider("Flow", 0, 10, 5)

lost_vac = st.slider("Lost Vacation", 0, 10, 0)
live_vision = st.slider("Live Vision", 0, 10, 5)
supporting = st.slider("Supporting Others", 0, 10, 4)
awards = st.slider("Personal Awards", 0, 10, 1)
donation = st.slider("Donation", 0, 10, 2)
achievement = st.slider("Achievement", 0, 10, 3)
core = st.slider("Core Circle", 0, 10, 5)
social = st.slider("Social Network", 0, 10, 5)
weekly_med = st.slider("Weekly Meditation", 0, 10, 2)
todo = st.slider("Todo Completed", 0, 10, 4)
bmi = st.slider("BMI Range", 0, 10, 2)
income = st.slider("Sufficient Income", 0, 10, 4)
places = st.slider("Places Visited", 0, 10, 3)
shouting = st.slider("Daily Shouting", 0, 10, 0)
passion = st.slider("Time for Passion", 0, 10, 3)


# ================================
# ENSAMBLAR INPUT
# ================================
user_input = {
    "GENDER": gender,
    "AGE": age,
    "DAILY_STRESS": stress,
    "SLEEP_HOURS": sleep,
    "DAILY_STEPS": steps,
    "FRUITS_VEGGIES": fruits,
    "FLOW": flow,
    "LOST_VACATION": lost_vac,
    "LIVE_VISION": live_vision,
    "SUPPORTING_OTHERS": supporting,
    "PERSONAL_AWARDS": awards,
    "DONATION": donation,
    "ACHIEVEMENT": achievement,
    "CORE_CIRCLE": core,
    "SOCIAL_NETWORK": social,
    "WEEKLY_MEDITATION": weekly_med,
    "TODO_COMPLETED": todo,
    "BMI_RANGE": bmi,
    "SUFFICIENT_INCOME": income,
    "PLACES_VISITED": places,
    "DAILY_SHOUTING": shouting,
    "TIME_FOR_PASSION": passion
}


# ================================
# PREPROCESS
# ================================
def preprocess_raw(df_raw, schema):
    df = pd.DataFrame(df_raw)

    # OHE automático por pipeline → NO aplicar get_dummies
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string").str.strip()

    for col in schema:
        if col not in df.columns:
            df[col] = np.nan

    return df[schema]


# ================================
# PREDICCIÓN
# ================================
if st.button("🔮 Predecir Score"):
    df_clean = preprocess_raw([user_input], schema_cols)

    pred = PIPE.predict(df_clean)[0]
    pred = float(np.clip(pred, lower, upper))

    st.success(f"🎯 Work-Life Balance Score: **{pred:.2f}**")
    st.write("📘 Datos procesados:")
    st.dataframe(df_clean)
