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

# Cambiar rango a 0–1000
lower = 0
upper = 1000

schema_cols = INPUT_SCHEMA["columns"]

st.title("🔮 Predictor de Equilibrio Vida–Trabajo")
st.write("Completa todos los campos para obtener una predicción:")

# ===============================
# SECCIÓN: VARIABLES CATEGÓRICAS
# ===============================
st.header("🧩 Información Personal")

# GENDER: 'Female'
gender = st.selectbox("Género:", ["Female", "Male"], index=0)

# AGE: '36 to 50'
age = st.selectbox(
    "Grupo de edad:",
    ["Less than 20", "21 to 35", "36 to 50", "51 or more"],
    index=2  # '36 to 50'
)

# DAILY_STRESS: '5' → slider en 5
daily_stress = st.slider("Estrés diario (0–5):", 0, 5, 5)

# ===============================
# SECCIÓN: HÁBITOS Y ESTILO DE VIDA
# ===============================
st.header("🏃 Hábitos y Estilo de Vida")

# SLEEP_HOURS: 6
sleep_hours = st.number_input("Horas de sueño por día(1-10)", 1, 10, 6)

# DAILY_STEPS: 10
daily_steps = st.number_input("Pasos diarios(1-10)", 1, 10, 10)

# En el ejemplo no hay valor para estos → los dejamos en 0 por defecto
physical_activity = st.number_input("Actividad física semanal (horas)", 0.0, 40.0, 0.0)
hydration = st.number_input("Hidratación (litros por día)", 0.0, 6.0, 0.0)
screen_time = st.number_input("Horas frente a pantallas por día", 0.0, 16.0, 0.0)

# WEEKLY_MEDITATION: 5
weekly_meditation = st.slider("Meditación (veces por semana 0-10)", 0, 10, 5)

# TIME_FOR_PASSION: 1
time_for_passion = st.slider("Tiempo para proyectos personales (0-10)", 0, 10, 1)

# ===============================
# SECCIÓN: RELACIONES SOCIALES
# ===============================
st.header("🤝 Relaciones Sociales")

# FRUITS_VEGGIES: 4
fruits = st.slider("Porciones de frutas y verduras(0-5)", 0, 5, 4)

# PLACES_VISITED: 4
places = st.slider("Lugares visitados por mes(0-10)", 0, 10, 4)

# CORE_CIRCLE: 6
core_circle = st.slider("Círculo cercano (0-10)", 0, 10, 6)

# SUPPORTING_OTHERS: 5
supporting_others = st.slider("Apoyo a otros (0–10)", 1, 10, 5)

# SOCIAL_NETWORK: 5
social_network = st.slider("Red social (1–10)", 0, 10, 5)

# ===============================
# SECCIÓN: LOGROS / PRODUCTIVIDAD
# ===============================
st.header("🏆 Logros y Productividad")

# ACHIEVEMENT: 0
achievement = st.slider("Logro personal (0–10)", 0, 10, 0)

# DONATION: 5
donation = st.slider("Donaciones por mes(0-5)", 0, 5, 5)

# BMI_RANGE: 2
bmi_range = st.slider("Categoría de IMC (1–2)", 1, 2, 2)

# TODO_COMPLETED: 4
todo_completed = st.slider("Tareas completadas diariamente (0–10)", 0, 10, 4)

# FLOW: 2
flow = st.slider("Estado de flow (0–10)", 0, 10, 2)

# LOST_VACATION: 5
lost_vacation = st.slider("Días de vacaciones perdidos(0-10)", 0, 10, 5)

# DAILY_SHOUTING: 1
daily_shouting = st.slider("Frecuencia de gritos diarios (0-10)", 0, 10, 1)

# SUFFICIENT_INCOME: 2
sufficient_income = st.slider("Satisfacción con los ingresos (1–2)", 1, 2, 2)

# PERSONAL_AWARDS: 5
personal_awards = st.slider("Premios personales (0–10)", 0, 10, 5)

# LIVE_VISION: 2
live_vision = st.slider("Claridad de visión de vida (0-10)", 0, 10, 2)

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
        "Predicción": pred_final,
        **user_input
    })

    st.success(f"🎯 Puntaje estimado de Equilibrio Vida–Trabajo: **{pred_final:.2f}**")
    # Probabilidad basada en el rango 0–1000
    prob = pred_final / 1000
    st.info(f"🔢 Probabilidad estimada de bienestar: **{prob*100:.2f}%**")


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
