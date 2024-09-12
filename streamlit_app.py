import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline
import os

# Paths to default CSV files
DEFAULT_CSV_PATH = 'artifacts/test.csv'
TRAIN_CSV_PATH = 'artifacts/train.csv'  # Path to the training dataset

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

    # Step 1: Load the training dataset
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # Convert 'Time' column in training data to human-readable format if it exists
    if 'Time' in train_df.columns:
        train_df['Time'] = pd.to_datetime(train_df['Time'], unit='s')  # Assuming Unix timestamps
        train_df.set_index('Time', inplace=True)

    # Show the first few rows of the training dataset with human-readable time
    st.write("Training dataset with human-readable time:")
    st.write(train_df.head())

    # Create a prediction pipeline object
    predictor = PredictionPipeline()

    # Predict anomalies
    results = predictor.predict(df)

    # Check if any anomalies were found
    if results["anomalies"].any():
        st.error("Anomalies found!")
        st.write("Anomalous data points:")
        st.write(results["anomalous_data"])

        # Step 2: Plot the anomalous data points against the training data for each sensor
        # sensors = ['Temperature', 'Humidity', 'Air Quality', 'Light', 'Loudness']

        # for sensor in sensors:
        #     plt.figure(figsize=(10, 6))

        #     # Plot training data for the sensor
        #     plt.plot(train_df.index, train_df[sensor], label=f'{sensor} Training Data', color='blue')

        #     # Plot test data (anomalous points)
        #     anomaly_indices = results['anomalous_data'].index
        #     plt.scatter(anomaly_indices, df.loc[anomaly_indices, sensor], color='red', label=f'Anomalous {sensor}', marker='o')

        #     # Set titles and labels
        #     plt.title(f'{sensor} Training Data with Anomalies from Test Data')
        #     plt.xlabel('Time')
        #     plt.ylabel(sensor)
        #     plt.legend()

        #     # Display the plot in Streamlit
        #     st.pyplot(plt)

        # Optionally, allow the user to download the anomalous data
        st.download_button(
            label="Download Anomalous Data",
            data=results["anomalous_data"].to_csv(index=False),
            file_name="anomalous_data.csv",
            mime="text/csv"
        )
    else:
        st.success("No anomalies found!")
