# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import load_config, load_data

st.set_page_config(layout="wide", page_title="Mental Distress Detection Dashboard")

def run_dashboard():
    config = load_config()
    dashboard_config = config['dashboard']
    trends_output_path = dashboard_config['input_data_for_dashboard']

    st.title("Mental Distress Detection Dashboard")

    if not os.path.exists(trends_output_path):
        st.error(f"Error: Data for dashboard not found at '{trends_output_path}'.")
        st.write("Please run the main pipeline (`python main.py`) first to generate the necessary data.")
        return

    df = load_data(trends_output_path)

    st.subheader("Distress Probability Over Time")
    fig_prob = px.line(df, y='distress_probability', title='Raw Distress Probability')
    st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("Smoothed Distress Trend")
    fig_trend = px.line(df, y='smoothed_trend', title='Smoothed Distress Trend (Moving Average)')
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Alerts Triggered")
    fig_alerts = px.area(df, y=['is_alert', 'triggered_alert'], title='Alerts (Is Alert / Triggered Alert)',
                         color_discrete_map={'is_alert': 'blue', 'triggered_alert': 'red'})
    st.plotly_chart(fig_alerts, use_container_width=True)

    st.subheader("Raw Data with Predictions")
    st.dataframe(df.head())

if __name__ == '__main__':
    run_dashboard()
