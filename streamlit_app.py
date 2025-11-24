st.subheader("Completa los datos")

# CATEGORÍAS QUE APRENDIÓ TU MODELO
gender = st.selectbox("Gender:", ["Female", "Male"])

age = st.selectbox("Age group:", [
    "Less than 20",
    "21 to 35",
    "36 to 50",
    "51 or more"
])

daily_stress = st.selectbox("Daily Stress (original categories):", [
    "0", "1", "1/1/00", "2", "3", "4", "5"
])

# VARIABLES NUMÉRICAS
sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
steps = st.number_input("Daily Steps", 0.0, 30000.0, 5000.0)
exercise = st.number_input("Physical Activity", 0.0, 20.0, 3.0)
water = st.number_input("Hydration (Liters)", 0.0, 5.0, 2.0)
screen = st.number_input("Screen Time (Hours)", 0.0, 12.0, 4.0)

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
