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
INPUT_SCHEMA = json.load(open(os.path.join(ART_DIR, "input_schema.json")))
POLICY = json.load(open(os.path.join(ART_DIR, "decision_policy.json")))

schema_cols = INPUT_SCHEMA["columns"]
lower = POLICY["lower"]
upper = POLICY["upper"]

st.title("💼 Predicción Work-Life Balance (WLB)")
st.subheader("Completa los datos del usuario")


# ================================
# VARIABLES DEL DATASET REAL
# ================================

gender = st.selectbox("Gender:", ["Male", "Female"])

age = st.selectbox(
    "Age group:",
    ["Less than 20", "21 to 35", "36 to 50", "51 or more"]  # ← CATEGORÍAS REALES
)

stress = st.selectbox("Daily Stress (0–5):", [0,1,2,3,4,5])

sleep = st.number_input("Sleep Hours:", min_value=0.0, max_value=12.0, value=7.0)

steps = st.number_input("Daily Steps:", min_value=0, max_value=50000, value=7000)

fruits = st.number_input("Fruits & Veggies Intake:", min_value=0, max_value=20, value=3)
flow = st.number_input("Flow:", min_value=0, max_value=10, value=5)

lost_vac = st.number_input("Lost Vacation:", 0, 10, 0)
live_vision = st.number_input("Live Vision:", 0, 10, 5)
supporting = st.number_input("Supporting Others:", 0, 10, 4)
awards = st.number_input("Personal Awards:", 0, 10, 1)
donation = st.number_input("Donation:", 0, 10, 2)
achievement = st.number_input("Achievement:", 0, 10, 3)
core = st.number_input("Core Circle:", 0, 10, 5)
social = st.number_input("Social Network:", 0, 10, 5)
weekly_med = st.number_input("Weekly Meditation:", 0, 10, 2)
todo = st.number_input("Todo Completed:", 0, 10, 4)
bmi = st.number_input("BMI Range:", 0, 10, 2)
income = st.number_input("Sufficient Income:", 0, 10, 4)
places = st.number_input("Places Visited:", 0, 10, 3)
shouting = st.number_input("Daily Shouting:", 0, 10, 0)
passion = st.number_input("Time for Passion:", 0, 10, 3)


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
# PREPROCESS PARA INFERENCIA
# ================================
def preprocess(df_raw, schema_cols):
    df = pd.DataFrame(df_raw)

    # (NO get_dummies; pipeline hace OHE)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string").str.strip()

    # Agregar columnas faltantes como NaN
    for col in schema_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ordenar columnas
    return df[schema_cols]


# ================================
# PREDICCIÓN
# ================================
if st.button("🔮 Predecir Score"):
    df_clean = preprocess([user_input], schema_cols)

    pred_raw = PIPE.predict(df_clean)[0]
    pred_final = float(np.clip(pred_raw, lower, upper))

    st.success(f"🎯 Work-Life Balance Score: **{pred_final:.2f}**")

    st.write("📘 Datos procesados:")
    st.dataframe(df_clean)
