import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from sklearn.preprocessing import MinMaxScaler

from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTRainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    # Helper function to create sequences
    def create_sequences(self, data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            # Step 1: Read the dataset
            df = pd.read_csv('notebooks/data/dataset_final.csv')
            logging.info('Read the dataset as dataframe')

            # Step 2: Set 'Time' column as index
            if 'Time' in df.columns:
                logging.info("Converting 'Time' column to human-readable format.")
                df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Assuming Unix timestamps
                df.set_index('Time', inplace=True)

            # Step 3: Normalize the data using MinMaxScaler
            scaler = MinMaxScaler()
            df_normalized = scaler.fit_transform(df.values)
            logging.info('Data normalization complete.')

            # Step 4: Define the sequence length
            sequence_length = 10

            # Step 5: Create sequences of normalized data
            X_data = self.create_sequences(df_normalized, sequence_length)
            logging.info(f'Sequences of length {sequence_length} created. Shape: {X_data.shape}')

            # Step 6: Split the normalized sequences into training and validation sets
            X_train, X_test = train_test_split(X_data, test_size=0.2, shuffle=True, random_state=42)
            logging.info(f'Train-test split complete. X_train shape: {X_train.shape}, X_val shape: {X_test.shape}')
            
            # Step 7: Save the train and validation datasets
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            np.save(self.ingestion_config.train_data_path, X_train)  # Save X_train as .npy file
            np.save(self.ingestion_config.test_data_path, X_test)  # Save X_val as .npy file
            logging.info("Ingestion complete and datasets saved.")

            return (X_train, X_test)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    modeltrainer = ModelTRainer()
    print(modeltrainer.initiate_model_trainer(train_data, test_data))