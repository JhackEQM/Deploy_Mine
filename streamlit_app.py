import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --------------------------
# Cargar artefactos
# --------------------------
ART_DIR = "artefactos"

PIPE_PATH = os.path.join(ART_DIR, "pipeline_LR.joblib")  # o pipeline_RG.joblib
SCHEMA_PATH = os.path.join(ART_DIR, "input_schema.json")
POLICY_PATH = os.path.join(ART_DIR, "decision_policy.json")

PIPE = joblib.load(PIPE_PATH)
INPUT_SCHEMA = json.load(open(SCHEMA_PATH, "r"))
POLICY = json.load(open(POLICY_PATH, "r"))

schema_cols = INPUT_SCHEMA["columns"]
lower = POLICY["lower"]
upper = POLICY["upper"]

st.title("🔮 Predicción Work-Life Balance")

# --------------------------
# Entrada de datos del usuario
# --------------------------

st.subheader("Completa los datos")

gender = st.selectbox("Gender:", ["Male", "Female"])
age = st.selectbox("Age group:", ["18-25", "26-33", "34-41", "42-49", "50-57", "58+"])
stress = st.selectbox("Daily Stress:", ["Low", "Medium", "High"])

sleep_hours = st.number_input("Hours of Sleep per Day", 0.0, 12.0, 7.0)
exercise = st.number_input("Exercise Hours Per Week", 0.0, 20.0, 3.0)
water = st.number_input("Daily Water Intake (Liters)", 0.0, 5.0, 2.0)
screen = st.number_input("Daily Screen Time (Hours)", 0.0, 12.0, 4.0)

# Crear el diccionario original (sin dummificar)
user_input = {
    "GENDER": gender,
    "AGE": age,
    "DAILY_STRESS": stress,
    "SLEEP_HOURS": sleep_hours,
    "PHYSICAL_ACTIVITY": exercise,
    "HYDRATION": water,
    "SCREEN_TIME": screen,
}

# --------------------------
# Función: procesamiento igual al entrenamiento
# --------------------------
def preprocess(df_raw: pd.DataFrame, schema_cols: list):
    df_proc = pd.get_dummies(df_raw, drop_first=True)

    # columnas faltantes → ponerlas en 0
    for col in schema_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0

    # columnas extras → eliminarlas
    df_proc = df_proc[schema_cols]

    return df_proc

# --------------------------
# Realizar predicción
# --------------------------
if st.button("🔮 Predecir"):
    df_raw = pd.DataFrame([user_input])
    df_ready = preprocess(df_raw, schema_cols)

    pred_raw = PIPE.predict(df_ready)[0]
    pred_final = float(np.clip(pred_raw, lower, upper))

    st.success(f"🎯 Predicción Work-Life Balance Score: **{pred_final:.2f}**")

    st.write("📘 Detalles de la entrada procesada:")
    st.dataframe(df_ready)
