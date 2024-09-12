import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline
import os

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
        plt.title(f'{col} Sensor Data with Anomalies')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        
        # Display the plot in Streamlit
        st.pyplot(plt)

# Paths to default CSV files
DEFAULT_CSV_PATH = 'artifacts/test.csv'
DATA_CSV_PATH = 'artifacts/data.csv'  # Path to the raw dataset

# Title of the Streamlit app
st.title("Anomaly Detection in IoT Sensor Data")

# Section to upload a CSV file
st.subheader("Upload a CSV File or Use Default")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Checkbox for using default dataset
use_default = st.checkbox("Use default dataset(test set from the source repo) instead of uploading")

# Button for user confirmation and starting prediction
if st.button("Start Prediction"):
    # Load the CSV file
    if uploaded_file is not None and not use_default:
        # If the user uploads a file, load it into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        # If no file is uploaded or the user selects the checkbox, use the default file
        st.info(f"Using default file: {DEFAULT_CSV_PATH}")
        df = pd.read_csv(DEFAULT_CSV_PATH)

    # Convert 'Time' column to human-readable format if it exists
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Assuming Unix timestamps in seconds
        df.set_index('Time', inplace=True)

    # Show the first few rows of the dataset with human-readable time
    st.write("Sample rows from chosen dataset:")
    st.write(df.head())

    # Step 1: Load the raw dataset
    raw_df = pd.read_csv(DATA_CSV_PATH)

    # Convert 'Time' column in  data to human-readable format if it exists
    if 'Time' in raw_df.columns:
        raw_df['Time'] = pd.to_datetime(raw_df['Time'], unit='s')  # Assuming Unix timestamps
        raw_df.set_index('Time', inplace=True)
    st.write("Sample rows from training dataset:")
    st.write(raw_df.head())
    # Create a prediction pipeline object
    predictor = PredictionPipeline()

    # Predict anomalies
    results = predictor.predict(df)
    
    # Check if any anomalies were found
    if results["anomalies"].any():
        st.error("Anomalies found!")
        st.write("Anomalous data points:")
        st.write(results["anomalous_data"])

       # Plot the training data with anomalies overlaid
        plot_train_data_with_anomalies(raw_df, results["anomalous_data"])

        # Optionally, allow the user to download the anomalous data
        st.download_button(
            label="Download Anomalous Data",
            data=results["anomalous_data"].to_csv(index=False),
            file_name="anomalous_data.csv",
            mime="text/csv"
        )
    else:
        st.success("No anomalies found!")
