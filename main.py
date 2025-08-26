import streamlit as st
import pandas as pd
import joblib

# Загружаем пайплайн
model = joblib.load("rf_pipeline.pkl")

st.title("Предсказания на модели RandomForest")
st.write("Введите значения признаков для предсказания риска сердечно-сосудистого заболевания:")

input_data = {}

# Числовые признаки с пояснениями
input_data['Age'] = st.number_input("Age", min_value=0, max_value=120, value=50)
st.caption("Возраст пациента в годах")

input_data['RestingBP'] = st.number_input("RestingBP", min_value=50, max_value=250, value=120)
st.caption("Артериальное давление в состоянии покоя (мм рт. ст.)")

input_data['Cholesterol'] = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
st.caption("Уровень холестерина (мг/дл)")

input_data['FastingBS'] = st.number_input("FastingBS", min_value=0, max_value=1, value=0)
st.caption("Глюкоза натощак > 120 мг/дл: 1 - да, 0 - нет")

input_data['MaxHR'] = st.number_input("MaxHR", min_value=60, max_value=250, value=150)
st.caption("Максимальная частота сердечных сокращений, достигнутая при нагрузке")

input_data['Oldpeak'] = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
st.caption("Снижение ST сегмента при нагрузке относительно покоя")

# Категориальные признаки с пояснениями
input_data['Sex'] = st.selectbox("Sex", options=["M", "F"])
st.caption("Пол пациента: M - мужчина, F - женщина")

input_data['ChestPainType'] = st.selectbox("ChestPainType", options=["TA", "ATA", "NAP", "ASY"])
st.caption("Тип боли в груди: TA - типичная ангина, ATA - атипичная ангина, NAP - неангинальная боль, ASY - бессимптомная")

input_data['RestingECG'] = st.selectbox("RestingECG", options=["Normal", "ST", "LVH"])
st.caption("ЭКГ в состоянии покоя: Normal - нормальная, ST - отклонения, LVH - гипертрофия левого желудочка")

input_data['ExerciseAngina'] = st.selectbox("ExerciseAngina", options=["Y", "N"])
st.caption("Боль при нагрузке: Y - да, N - нет")

input_data['ST_Slope'] = st.selectbox("ST_Slope", options=["Up", "Flat", "Down"])
st.caption("Наклон сегмента ST при нагрузке: Up - восходящий, Flat - ровный, Down - нисходящий")

# Предсказание
if st.button("Сделать предсказание"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    st.write(f"**Предсказанный класс:** {prediction[0]}")
    st.caption("1 - высокий риск заболевания, 0 - низкий риск")