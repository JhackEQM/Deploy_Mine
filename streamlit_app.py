import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =========================================
# 1) Cargar artefactos
# =========================================
ART_DIR = "artefactos"

PIPE_PATH   = os.path.join(ART_DIR, "pipeline_RG.joblib")
SCHEMA_PATH = os.path.join(ART_DIR, "input_schema.json")
POLICY_PATH = os.path.join(ART_DIR, "decision_policy.json")

PIPE = joblib.load(PIPE_PATH)
INPUT_SCHEMA = json.load(open(SCHEMA_PATH, "r"))
POLICY = json.load(open(POLICY_PATH, "r"))

schema_cols = INPUT_SCHEMA["columns"]
dtypes      = INPUT_SCHEMA["dtypes"]

lower = POLICY.get("lower", None)
upper = POLICY.get("upper", None)

# =========================================
# 2) UI
# =========================================
st.title("🔮 Work-Life Balance Predictor")

st.subheader("Completa los datos del usuario")

# 🟦 CATEGORÍAS QUE APRENDIÓ TU MODELO (100% REALES)
gender = st.selectbox("Gender:", ["Female", "Male"])

age = st.selectbox("Age group:", [
    "Less than 20",
    "21 to 35",
    "36 to 50",
    "51 or more"
])

daily_stress = st.selectbox("Daily Stress (categorías originales):", [
    "0", "1", "1/1/00", "2", "3", "4", "5"
])

# 🟩 VARIABLES NUMÉRICAS
sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
steps = st.number_input("Daily Steps", 0.0, 30000.0, 5000.0)
exercise = st.number_input("Physical Activity (hours/week)", 0.0, 20.0, 3.0)
water = st.number_input("Hydration (Liters/day)", 0.0, 5.0, 2.0)
screen = st.number_input("Screen Time (Hours/day)", 0.0, 12.0, 4.0)

# Diccionario EXACTO según schema del entrenamiento
user_input = {
    "GENDER": gender,
    "AGE": age,
    "DAILY_STRESS": daily_stress,
    "SLEEP_HOURS": sleep_hours,
    "DAILY_STEPS": steps,
    "PHYSICAL_ACTIVITY": exercise,
    "HYDRATION": water,
    "SCREEN_TIME": screen,
}

# =========================================
# 3) Preprocesamiento igual al entrenamiento
# =========================================
def coerce_and_align(df_raw, schema_cols, dtypes):
    df = df_raw.copy()

    # Agregar columnas faltantes
    for c in schema_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Convertir tipos según el esquema
    for c in schema_cols:
        dtype = dtypes[c]
        if dtype.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string").str.strip()

    return df[schema_cols]

# =========================================
# 4) Predicción
# =========================================
if st.button("🔮 Predecir"):
    df_raw = pd.DataFrame([user_input])

    df_clean = coerce_and_align(df_raw, schema_cols, dtypes)

    pred_raw = PIPE.predict(df_clean)[0]

    # Aplicar política
    pred_final = float(np.clip(pred_raw, lower, upper))

    st.success(f"🎯 Predicción Work-Life Balance Score: **{pred_final:.2f}**")

    st.write("📘 Datos enviados al modelo (ya alineados):")
    st.dataframe(df_clean)
