import streamlit as st
import pandas as pd
import numpy as np
import json, joblib
import os

# ============================
# 1) Cargar artefactos
# ============================

ART_DIR = "artefactos"

PIPE_PATH = os.path.join(ART_DIR, "pipeline_MLP.joblib")
SCHEMA_PATH = os.path.join(ART_DIR, "input_schema.json")
POLICY_PATH = os.path.join(ART_DIR, "decision_policy.json")

PIPE = joblib.load(PIPE_PATH)
INPUT_SCHEMA = json.load(open(SCHEMA_PATH, "r", encoding="utf-8"))
POLICY = json.load(open(POLICY_PATH, "r", encoding="utf-8"))

FEATURES = INPUT_SCHEMA["columns"]
DTYPES = INPUT_SCHEMA["dtypes"]


# ============================
# 2) Funciones auxiliares
# ============================

def _coerce_and_align(df):
    """Alinea columnas y convierte tipos según el esquema de entrenamiento."""
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan

        t = str(DTYPES[c]).lower()

        if t.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string").str.strip()

    return df[FEATURES]


def predict_single(record: dict):
    df = pd.DataFrame([record])
    df = _coerce_and_align(df)
    
    yhat = PIPE.predict(df)

    lower = POLICY.get("lower", float(np.min(yhat)))
    upper = POLICY.get("upper", float(np.max(yhat)))
    
    ypp = float(np.clip(yhat[0], lower, upper))
    return ypp


# ============================
# 3) Interfaz Streamlit
# ============================

st.title("🧠 Predicción de Work-Life Balance")
st.write("Ingrese los parámetros de estilo de vida para predecir el puntaje de bienestar laboral/personal.")

user_inputs = {}

# Crear widgets según el esquema
for col in FEATURES:
    t = str(DTYPES[col]).lower()

    if t.startswith("int") or t.startswith("float"):
        user_inputs[col] = st.number_input(col, value=0.0)
    else:
        user_inputs[col] = st.text_input(col, "")

# ============================
# 4) Botón para predecir
# ============================

if st.button("Predecir"):
    pred = predict_single(user_inputs)
    st.success(f"🎯 Predicción estimada del Work-Life Balance Score: **{pred:.2f}**")
