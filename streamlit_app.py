import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =========================================
# Inicializar historial persistente
# =========================================
if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================================
# Cargar artefactos
# =========================================
ART_DIR = "artefactos"

PIPE_PATH = os.path.join(ART_DIR, "pipeline_RG.joblib")
SCHEMA_PATH = os.path.join(ART_DIR, "input_schema.json")
POLICY_PATH = os.path.join(ART_DIR, "decision_policy.json")

PIPE = joblib.load(PIPE_PATH)
INPUT_SCHEMA = json.load(open(SCHEMA_PATH, "r"))
POLICY = json.load(open(POLICY_PATH, "r"))

schema_cols = INPUT_SCHEMA["columns"]
lower = POLICY["lower"]
upper = POLICY["upper"]

st.title("🔮 Work-Life Balance Predictor")
st.write("Complete all the inputs below:")

# ===============================
# SECCIÓN: VARIABLES CATEGÓRICAS
# ===============================
st.header("🧩 Información Personal")

gender = st.selectbox("Gender:", ["Female", "Male"])

age = st.selectbox("Age group:", [
    "Less than 20",
    "21 to 35",
    "36 to 50",
    "51 or more"
])

daily_stress = st.slider("Daily Stress (0–5):", 0, 5, 2)

# ===============================
# SECCIÓN: HÁBITOS Y ESTILO DE VIDA
# ===============================
st.header("🏃 Hábitos y Estilo de Vida")

sleep_hours = st.number_input("Sleep Hours per Day", 0.0, 12.0, 7.0)
daily_steps = st.number_input("Daily Steps", 0, 30000, 5000)
physical_activity = st.number_input("Weekly Physical Activity (hours)", 0.0, 40.0, 5.0)
hydration = st.number_input("Hydration (liters per day)", 0.0, 6.0, 2.0)
screen_time = st.number_input("Screen Time (hours per day)", 0.0, 16.0, 4.0)

weekly_meditation = st.slider("Meditation (times per week)", 0, 14, 2)
time_for_passion = st.slider("Time for Passion Projects (1–5)", 1, 5, 3)

# ===============================
# SECCIÓN: RELACIONES SOCIALES
# ===============================
st.header("🤝 Relaciones Sociales")

fruits = st.slider("Fruits & Veggies Servings", 0, 10, 4)
places = st.slider("Places Visited per Month", 0, 20, 3)
core_circle = st.slider("Core Circle (Close friends)", 0, 20, 5)
supporting_others = st.slider("Supporting Others (1–5)", 1, 5, 3)
social_network = st.slider("Social Network Strength (1–5)", 1, 5, 3)

# ===============================
# SECCIÓN: LOGROS / PRODUCTIVIDAD
# ===============================
st.header("🏆 Logros y Productividad")

achievement = st.slider("Achievement (0–5)", 0, 5, 2)
donation = st.slider("Donations per Month", 0, 10, 1)
bmi_range = st.slider("BMI Range Category (1–5)", 1, 5, 2)
todo_completed = st.slider("Daily TODO Completion (1–5)", 1, 5, 3)
flow = st.slider("Flow State Frequency (1–5)", 1, 5, 2)
lost_vacation = st.slider("Lost Vacation Days", 0, 60, 5)
daily_shouting = st.slider("Daily Shouting Frequency (1–5)", 1, 5, 1)
sufficient_income = st.slider("Income Satisfaction (1–5)", 1, 5, 3)
personal_awards = st.slider("Personal Awards (0–10)", 0, 10, 1)
live_vision = st.slider("Life Vision Clarity (1–5)", 1, 5, 3)

# ================================================
# Crear diccionario EXACTO que el modelo espera
# ================================================
user_input = {
    "FRUITS_VEGGIES": fruits,
    "DAILY_STRESS": daily_stress,
    "PLACES_VISITED": places,
    "CORE_CIRCLE": core_circle,
    "SUPPORTING_OTHERS": supporting_others,
    "SOCIAL_NETWORK": social_network,
    "ACHIEVEMENT": achievement,
    "DONATION": donation,
    "BMI_RANGE": bmi_range,
    "TODO_COMPLETED": todo_completed,
    "FLOW": flow,
    "DAILY_STEPS": daily_steps,
    "LIVE_VISION": live_vision,
    "SLEEP_HOURS": sleep_hours,
    "LOST_VACATION": lost_vacation,
    "DAILY_SHOUTING": daily_shouting,
    "SUFFICIENT_INCOME": sufficient_income,
    "PERSONAL_AWARDS": personal_awards,
    "TIME_FOR_PASSION": time_for_passion,
    "WEEKLY_MEDITATION": weekly_meditation,
    "AGE": age,
    "GENDER": gender,
    "PHYSICAL_ACTIVITY": physical_activity,
    "HYDRATION": hydration,
    "SCREEN_TIME": screen_time
}

# =========================================
# Alineación EXACTA al schema
# =========================================
def align(df_raw, schema):
    df = df_raw.copy()
    cols = schema["columns"]
    dtypes = schema["dtypes"]

    for col in cols:
        if col not in df:
            df[col] = np.nan

    for col in cols:
        t = str(dtypes[col])
        if "int" in t or "float" in t:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string")

    return df[cols]

# =========================================
# PREDICCIÓN
# =========================================
if st.button("🔮 Predecir"):
    df_raw = pd.DataFrame([user_input])
    df_clean = align(df_raw, INPUT_SCHEMA)

    pred_raw = PIPE.predict(df_clean)[0]
    pred_final = float(np.clip(pred_raw, lower, upper))

    # Guardar en historial
    st.session_state["history"].append({
        "prediction": pred_final,
        **user_input
    })

    st.success(f"🎯 Predicción Work-Life Balance Score: **{pred_final:.2f}**")

    st.write("📘 Entrada procesada:")
    st.dataframe(df_clean)

# =========================================
# Mostrar historial acumulado
# =========================================
st.header("📚 Historial de predicciones")

if len(st.session_state["history"]) > 0:
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df)
else:
    st.info("Aún no hay predicciones registradas.")
