import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object  # Assuming save_object and load_object are present for reusability
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass

@dataclass
class PredictConfig:
    model_file_path: str = os.path.join("artifacts", "model_new.keras")
    threshold_file_path: str = os.path.join("artifacts", "errors.txt")
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class PredictionPipeline:
    def __init__(self):
        self.config = PredictConfig()
    
    def load_threshold(self):
        """ Load the threshold value from the saved text file. """
        try:
            with open(self.config.threshold_file_path, 'r') as file:
                threshold = float(file.read().split(': ')[1])
            return threshold
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_input(self, df):
        """ Preprocess the input dataframe using the same scaler used during training. """
        try:
            logging.info("Loading preprocessor object.")
            preprocessor = load_object(self.config.preprocessor_file_path)

            # Transform the input DataFrame
            input_data = preprocessor.transform(df)

            # Assuming you need to create sequences of the data, similar to training
            sequence_length = 10  # Keep this consistent with training
            def create_sequences(data, sequence_length):
                sequences = []
                for i in range(len(data) - sequence_length + 1):
                    sequences.append(data[i:i+sequence_length])
                return np.array(sequences)
            
            input_sequences = create_sequences(input_data, sequence_length)
            return input_sequences

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df):
        """ Main function to predict anomalies on the custom input DataFrame. """
        try:
            # Step 1: Load the saved LSTM model
            logging.info(f"Loading model from {self.config.model_file_path}")
            model = load_model(self.config.model_file_path)

            # Step 2: Load the threshold
            logging.info(f"Loading threshold from {self.config.threshold_file_path}")
            threshold = self.load_threshold()

            # Step 3: Convert the 'time' column to human-readable format
            if 'Time' in df.columns:
                logging.info("Converting 'Time' column to human-readable format.")
                df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Assuming Unix timestamps
                df.set_index('Time', inplace=True)

            # Step 4: Preprocess the input dataframe
            logging.info("Preprocessing input data.")
            processed_data = self.preprocess_input(df)

            # Step 5: Predict using the model
            logging.info("Generating predictions on input data.")
            predictions = model.predict(processed_data)

            # Step 6: Calculate reconstruction error
            reconstruction_error = np.mean(np.square(processed_data - predictions), axis=(1, 2))

            # Step 7: Compare reconstruction error to threshold to detect anomalies
            logging.info("Detecting anomalies based on threshold.")
            anomalies = reconstruction_error > threshold

            # Step 8: Extract actual data points flagged as anomalies
            logging.info("Extracting anomalous data points.")
            anomaly_indices = np.where(anomalies)[0]
            anomaly_data = df.iloc[anomaly_indices]

            # Return results
            return {
                "reconstruction_error": reconstruction_error,
                "anomalies": anomalies,
                "anomalous_data": anomaly_data  # Return the actual anomalous data points
            }

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    df = pd.read_csv('artifacts/test.csv')
    predictor = PredictionPipeline()
    results = predictor.predict(df)

        # Access the anomalies
    anomalous_data = results["anomalous_data"]

    # Print the anomalies
    print("Anomalous data points:")
    print(anomalous_data)