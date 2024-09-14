import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline
import os
import time  # Import the time module for sleep
from src.utils import plot_train_data_with_anomalies, convert_and_set_time_index

# Custom CSS Styling
st.markdown("""
    <style>
    .title h1 {
        font-size: 36px;  /* Adjust the size to fit on one line */
        white-space: nowrap;  /* Prevent the title from wrapping */
    }
    .main {
        background-color: #f0f2f6;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        color: #003366;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
        color: #003366;
    }
    .sample-data-box {
        background-color: #e6f7ff;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #007acc;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Anomaly Detection Dashboard")
st.sidebar.subheader("Choose Your Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset from source repo", value=True)

# Main Title
st.markdown('<div class="title"><h1>Anomaly Detection in IoT Sensor Data</h1></div>', unsafe_allow_html=True)

# Sample Data Box
#st.markdown('<div class="sample-data-box">', unsafe_allow_html=True)
st.subheader("Sample Data Format")
st.markdown("""
**Default Dataset Description**: The default dataset is collected from a 25 m² room over a period of 24 hours with 2 people present. 

**Expected Data Format**:
- **Time**: Unix timestamp (seconds) or human readable format
- **Temperature**: Degrees Celsius (°C)
- **Humidity**: Percentage (%)
- **Air Quality**: Index or ppm (specific air quality measurement)
- **Light**: Lux (lx)
- **Loudness**: Decibels (dB)

Example:
| Time                      | Temperature (°C) | Humidity (%) | Air Quality | Light (lx) | Loudness (dB) |
| -------------             | ---------------- | ------------ | ----------- | ---------- | ------------- |
| 2021-06-15 18:21:46       | 37.94            | 28.94        | 75          | 644        | 106           |
| 2021-06-15 18:21:56       | 37.94            | 29.00        | 75          | 645        | 145           |
""")
st.markdown('</div>', unsafe_allow_html=True)

# Progress Bar
progress_bar = st.sidebar.progress(0)

# Button for user confirmation and starting prediction
if st.sidebar.button("Start Prediction"):
    # Load the CSV file
    if uploaded_file is not None and not use_default:
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            st.stop()
        else:
            df = pd.read_csv(uploaded_file)
            data_set_for_plot = df.copy()#pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
    else:
        st.info(f"Using default dataset from source")
        df = pd.read_csv('artifacts/data.csv')
        data_set_for_plot = pd.read_csv('artifacts/data.csv')

    # Convert 'Time' column to human-readable format
    df = convert_and_set_time_index(df)

    # Display the dataset
    st.subheader("Chosen Dataset Preview")
    st.write(df.head())

    # Load the raw dataset
    # raw_df = pd.read_csv('artifacts/data.csv')
    # raw_df = convert_and_set_time_index(raw_df)
    data_set_for_plot = convert_and_set_time_index(data_set_for_plot)

    # Create prediction pipeline
    predictor = PredictionPipeline()

    # Simulate progress
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.005)

    # Predict anomalies
    results = predictor.predict(df)
    if isinstance(results, str):
        st.error(results)  # Display the error message
    else:
        # Check if anomalies are found
        if results["anomalies"].any():
            st.error("Anomalies found!")
            st.write("Anomalous data points:")
            st.write(results["anomalous_data"])

            # Plot training data with anomalies
            plot_train_data_with_anomalies(data_set_for_plot, results["anomalous_data"])

            # Download button for anomalous data
            st.download_button(
                label="Download Anomalous Data",
                data=results["anomalous_data"].to_csv(index=False),
                file_name="anomalous_data.csv",
                mime="text/csv"
            )
        else:
            st.success("No anomalies found!")

    # Progress Completion
    progress_bar.empty()
