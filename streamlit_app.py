import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os

st.set_page_config(page_title="Work-Life Balance Predictor", page_icon="🔮")

# ------------------------------------------------------------
# 1. Cargar artefactos
# ------------------------------------------------------------
ART_DIR = "artefactos"

SCHEMA_PATH = os.path.join(ART_DIR, "input_schema.json")
POLICY_PATH = os.path.join(ART_DIR, "decision_policy.json")

with open(SCHEMA_PATH, "r") as f:
    INPUT_SCHEMA = json.load(f)

with open(POLICY_PATH, "r") as f:
    POLICY = json.load(f)

WINNER = POLICY["winner"]
PIPE_PATH = os.path.join(ART_DIR, f"pipeline_{WINNER}.joblib")
PIPE = joblib.load(PIPE_PATH)

schema_cols = INPUT_SCHEMA["columns"]
lower = POLICY["lower"]
upper = POLICY["upper"]

st.title("🔮 Predicción de Work-Life Balance Score")

# ------------------------------------------------------------
# 2. Entrada del usuario (adaptar a categorías reales del dataset)
# ------------------------------------------------------------

st.subheader("Completa los datos del usuario:")

gender = st.selectbox("Gender:", ["Male", "Female"])
age = st.selectbox("Age group:", ["1/1/00"])   # El dataset tenía valores raros
stress = st.selectbox("Daily Stress (1–5):", [1, 2, 3, 4, 5])

sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=7000)
fruits = st.number_input("Fruits & Veggies intake", min_value=0, max_value=10, value=3)
flow = st.number_input("Flow", min_value=0, max_value=10, value=5)

# Diccionario EXACTO como el dataset original
user_input = {
    "GENDER": gender,
    "AGE": age,
    "DAILY_STRESS": stress,
    "SLEEP_HOURS": sleep_hours,
    "DAILY_STEPS": daily_steps,
    "FRUITS_VEGGIES": fruits,
    "FLOW": flow,
}

# ------------------------------------------------------------
# 3. Preprocesamiento igual que en entrenamiento
# ------------------------------------------------------------
def preprocess_input(df_raw: pd.DataFrame, schema_cols):

    # One-Hot Encoding
    df_proc = pd.get_dummies(df_raw, drop_first=True)

    # Agregar columnas faltantes
    for col in schema_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0

    # Ordenar
    df_proc = df_proc[schema_cols]

    return df_proc


# ------------------------------------------------------------
# 4. Predicción
# ------------------------------------------------------------
if st.button("🔮 Predecir Score"):
    df_raw = pd.DataFrame([user_input])
    df_ready = preprocess_input(df_raw, schema_cols)

    pred_raw = PIPE.predict(df_ready)[0]
    pred_final = float(np.clip(pred_raw, lower, upper))

    st.success(f"🎯 Work-Life Balance Score: **{pred_final:.2f}**")

    with st.expander("Ver datos procesados"):
        st.dataframe(df_ready)
