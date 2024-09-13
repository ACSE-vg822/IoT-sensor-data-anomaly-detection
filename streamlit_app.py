import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline
import os
import time  # Import the time module for sleep

# Function to plot training data with anomalies
def plot_train_data_with_anomalies(df, anomalies_df):
    # Iterate over each sensor/column in the DataFrame
    for col in df.columns:
        # Create a new figure for each column
        plt.figure(figsize=(15, 8))  # Adjust the figure size as needed
        
        # Plot training data for the current column
        plt.plot(df.index, df[col], marker='.', linestyle='-', label=f'Training {col}')
        
        # Plot anomaly points (if any) for the current column
        if col in anomalies_df.columns:
            plt.scatter(anomalies_df.index, anomalies_df[col], color='red', marker='o', label=f'Anomalies in {col}', s=100, zorder=5)  # Set the size of the dots (s=100)
        
        # Set titles and labels
        plt.title(f'{col} Sensor Data with Anomalies', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(col, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Display the plot in Streamlit
        st.pyplot(plt)

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
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Anomaly Detection Dashboard")
st.sidebar.subheader("Choose Your Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset (test data from source repo)", value=True)

# Main Title
#st.title("Anomaly Detection in IoT Sensor Data")
st.markdown('<div class="title"><h1>Anomaly Detection in IoT Sensor Data</h1></div>', unsafe_allow_html=True)

# Progress Bar
progress_bar = st.sidebar.progress(0)

# Button for user confirmation and starting prediction
if st.sidebar.button("Start Prediction"):
    # Load the CSV file
    if uploaded_file is not None and not use_default:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info(f"Using default dataset from source")
        df = pd.read_csv('artifacts/test.csv')

    # Convert 'Time' column to human-readable format if it exists
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], unit='s')
        df.set_index('Time', inplace=True)

    # Display the dataset
    st.subheader("Chosen Dataset Preview")
    st.write(df.head())

    # Load the raw dataset
    raw_df = pd.read_csv('artifacts/data.csv')

    if 'Time' in raw_df.columns:
        raw_df['Time'] = pd.to_datetime(raw_df['Time'], unit='s')
        raw_df.set_index('Time', inplace=True)

    # Create prediction pipeline
    predictor = PredictionPipeline()

    # Simulate progress
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.05)  # Use time.sleep instead of st.sleep

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
            plot_train_data_with_anomalies(raw_df, results["anomalous_data"])

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
