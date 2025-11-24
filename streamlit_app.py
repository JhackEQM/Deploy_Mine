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

SCHEMA_PATH = os.path.join(ART_DIR, "input_schema.json")
POLICY_PATH = os.path.join(ART_DIR, "decision_policy.json")

INPUT_SCHEMA = json.load(open(SCHEMA_PATH, "r"))
POLICY = json.load(open(POLICY_PATH, "r"))

WINNER = POLICY["winner"]
PIPE = joblib.load(os.path.join(ART_DIR, f"pipeline_{WINNER}.joblib"))

schema_cols = INPUT_SCHEMA["columns"]
lower = POLICY["lower"]
upper = POLICY["upper"]


st.title("🔮 Predicción Work-Life Balance Score")
st.subheader("Completa los datos del usuario")


# --------------------------
# VARIABLES REALES DEL DATASET
# --------------------------

gender = st.selectbox("Gender:", ["Male", "Female"])
age = st.selectbox("Age group:", ["Less than 20", "21 to 35", "36 to 50", "51 or more"])
stress = st.selectbox("Daily Stress (1–5):", [0,1,2,3,4,5])

sleep = st.slider("Sleep Hours", 0, 12, 7)
steps = st.number_input("Daily Steps", 0, 30000, 7000)
fruits = st.slider("Fruits & Veggies Intake", 0, 10, 3)
flow = st.slider("Flow", 0, 10, 5)

lost_vacation = st.slider("Lost Vacation", 0,10,0)
live_vision = st.slider("Live Vision", 0,10,5)
supporting = st.slider("Supporting Others", 0,10,5)
personal_awards = st.slider("Personal Awards", 0,10,1)
donation = st.slider("Donation", 0,10,2)
achievement = st.slider("Achievement", 0,10,3)
core_circle = st.slider("Core Circle", 0,10,5)
social_network = st.slider("Social Network", 0,10,5)
weekly_med = st.slider("Weekly Meditation", 0,10,2)
todo_completed = st.slider("Todo Completed", 0,10,5)
bmi_range = st.slider("BMI Range", 0,10,2)
sufficient_income = st.slider("Sufficient Income", 0,10,3)
places_visited = st.slider("Places Visited", 0,10,4)
daily_shouting = st.slider("Daily Shouting", 0,10,0)

# --------------------------
# CREAR INPUT COMPLETO
# --------------------------

user_input = {
    "GENDER": gender,
    "AGE": age,
    "DAILY_STRESS": stress,
    "SLEEP_HOURS": sleep,
    "DAILY_STEPS": steps,
    "FRUITS_VEGGIES": fruits,
    "FLOW": flow,
    "LOST_VACATION": lost_vacation,
    "LIVE_VISION": live_vision,
    "SUPPORTING_OTHERS": supporting,
    "PERSONAL_AWARDS": personal_awards,
    "DONATION": donation,
    "ACHIEVEMENT": achievement,
    "CORE_CIRCLE": core_circle,
    "SOCIAL_NETWORK": social_network,
    "WEEKLY_MEDITATION": weekly_med,
    "TODO_COMPLETED": todo_completed,
    "BMI_RANGE": bmi_range,
    "SUFFICIENT_INCOME": sufficient_income,
    "PLACES_VISITED": places_visited,
    "DAILY_SHOUTING": daily_shouting,
}

# --------------------------
# FUNCIONES
# --------------------------

def preprocess(df_raw, schema_cols):
    df_proc = pd.get_dummies(df_raw, drop_first=True)

    for col in schema_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0

    df_proc = df_proc[schema_cols]
    return df_proc


# --------------------------
# PREDICCIÓN
# --------------------------

if st.button("🔮 Predecir Score"):
    df = pd.DataFrame([user_input])
    df_clean = preprocess(df, schema_cols)

    pred = PIPE.predict(df_clean)[0]
    pred = float(np.clip(pred, lower, upper))

    st.success(f"Score estimado: **{pred:.2f}**")
    st.write("📘 Entrada procesada:")
    st.dataframe(df_clean)
