import os
import sys
import streamlit as st

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

# Function to plot training data with anomalies    
def plot_train_data_with_anomalies(df, anomalies_df):
    # Iterate over each sensor/column in the DataFrame
    for col in df.columns:
        # Create a new figure for each column
        plt.figure(figsize=(25, 8))  # Adjust the figure size as needed
        
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

def convert_and_set_time_index(df, time_column='Time'):
    # Converts the 'Time' column to a human-readable format if it is not already,
    # and sets it as the index of the DataFrame
    if time_column in df.columns:
        logging.info(f"Checking if '{time_column}' column needs to be converted to human-readable format.")

        # Check if 'Time' is already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            logging.info(f"Converting '{time_column}' column to human-readable format.")
            df[time_column] = pd.to_datetime(df[time_column], unit='s')  # Convert to human-readable format

        # Set 'Time' as the index regardless of format
        df.set_index(time_column, inplace=True)
        logging.info(f"Set '{time_column}' column as the index.")
    
    return df
