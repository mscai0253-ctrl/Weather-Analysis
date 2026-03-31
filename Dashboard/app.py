import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocess import load_and_clean_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.predict import predict_temp

st.set_page_config(page_title="Weather Dashboard", layout="wide")


st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }

    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title"> Advanced Weather Dashboard</div>', unsafe_allow_html=True)


df = load_and_clean_data()


st.sidebar.header(" Filters")

city = st.sidebar.selectbox("Select City", df["City"].unique())
filtered_df = df[df["City"] == city]


col1, col2, col3 = st.columns(3)

col1.metric(" Avg Temperature", f"{filtered_df['temperature'].mean():.2f} °C")
col2.metric(" Avg Humidity", f"{filtered_df['Humidity_%'].mean():.2f}%")
col3.metric(" Avg Rainfall", f"{filtered_df['Precipitation_mm'].mean():.2f} mm")


st.subheader(" Temperature Trend")

fig1 = px.line(
    filtered_df,
    x="date",
    y="temperature",
    title=f"Temperature Over Time ({city})",
    markers=True
)
st.plotly_chart(fig1, use_container_width=True)


st.subheader(" Humidity vs Temperature")

fig2 = px.scatter(
    filtered_df,
    x="Humidity_%",
    y="temperature",
    color="temperature",
    size="Wind_Speed_km_h",
    title="Humidity vs Temperature"
)
st.plotly_chart(fig2, use_container_width=True)


st.subheader(" Weather Condition Distribution")

fig3 = px.pie(
    filtered_df,
    names="Condition",
    title="Weather Conditions"
)
st.plotly_chart(fig3)


models, X_test, y_test = train_models(df)


st.subheader(" Model Performance")

results = evaluate_models(models, X_test, y_test)

fig4 = px.bar(
    results,
    x="Model",
    y="R2 Score",
    color="Model",
    text_auto=True,
    title="Model Comparison (R² Score)"
)
st.plotly_chart(fig4, use_container_width=True)


st.subheader(" Predict Future Temperature")

col1, col2, col3 = st.columns(3)

day = col1.number_input("Day", 1, 31)
month = col2.number_input("Month", 1, 12)
year = col3.number_input("Year", 2000, 2100)

humidity = st.slider("Humidity (%)", 0, 100)
wind = st.slider("Wind Speed (km/h)", 0, 150)
precipitation = st.slider("Precipitation (mm)", 0.0, 500.0)

if st.button(" Predict Temperature"):
    result = predict_temp(day, month, year, humidity, wind, precipitation)
    
    st.success(f" Predicted Temperature: {result:.2f} °C")

    
    if result > 35:
        st.warning(" Hot Weather Expected!")
    elif result < 15:
        st.info(" Cold Weather Expected!")
    else:
        st.info(" Moderate Weather")
