import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained Linear Regression model
model = joblib.load('linear_regression_model.joblib')

# Function to predict the value based on input date
def predict_value(input_date):
    # Convert the input date to a numeric value (days since the training start date)
    X_input_num = (input_date - pd.to_datetime("2018-12-31")).days
    prediction = model.predict(np.array([[X_input_num]]))
    return prediction[0]

# Streamlit UI
st.title("📈 Regression Forecasting ")

# User inputs for start and end dates
start_date = st.date_input("Select Start Date", pd.to_datetime("2018-12-31"))
end_date = st.date_input("Select End Date", pd.to_datetime("2019-12-31"))

# Convert to pandas datetime format
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Ensure start date is before end date
if start_date >= end_date:
    st.error("End date must be after start date. Please adjust your selection.")
else:
    # Generate forecast dates based on user input
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    forecast_values = [predict_value(date) for date in forecast_dates]

    # Create a Plotly figure for the forecast trend
    fig = go.Figure()

    # Add forecasted values as a dashed line with markers
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        marker=dict(color="green"),
        line=dict(dash="dash", color="green")
    ))

    # Update layout for better visualization
    fig.update_layout(
        title="📊 Forecast Trend Based on Regression",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        template="plotly_white",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Show forecasted data table
    forecast_data = pd.DataFrame({"Date": forecast_dates, "Forecasted Value": forecast_values})
