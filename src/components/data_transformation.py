import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Temperature', 'Humidity', 'Air Quality', 'Light', 'Loudness']
            num_pipeline = Pipeline(
                steps =[
                ("scaler", MinMaxScaler())
                ]
            )

            logging.info("Numerical pipeline created")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)              
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Read train and test data completed")
                logging.info("Obtaining preprocessing object")

                # Process the 'Time' column for both train and test data
                def process_time_column(df):
                    df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Convert to datetime
                    df.set_index('Time', inplace=True)  # Set as index
                    logging.info("Processed 'Time' column for dataframe.")
                    return df

                # Apply the time processing to both train and test sets
                train_df = process_time_column(train_df)
                test_df = process_time_column(test_df)

                preprocessing_obj = self.get_data_transformer_object()
                # scaler = MinMaxScaler()

                logging.info("Applying transformations to train and test data")

                # Fit the preprocessor on training data and transform both train and test data
                input_features_train_transformed = preprocessing_obj.fit_transform(train_df)
                input_features_test_transformed = preprocessing_obj.transform(test_df)

                # Create sequences of data
                sequence_length = 10

                # Function to create sequences
                def create_sequences(data, sequence_length):
                    X = []
                    for i in range(len(data) - sequence_length + 1):
                        X.append(data[i:i + sequence_length])
                    return np.array(X)

                # Create sequences for training and testing data
                X_train = create_sequences(input_features_train_transformed, sequence_length)
                X_test = create_sequences(input_features_test_transformed, sequence_length)

                logging.info(f"Created sequences for train and test data with sequence length {sequence_length}.")

                logging.info("Data transformation completed.")

                # Save the preprocessing object for future use (e.g., in deployment)
                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessing_obj
                )

                logging.info(f"Saved preprocessing object.")

                # Return the transformed training and test data, along with the preprocessor path
                return (
                    X_train,  # Transformed training data (no target)
                    X_test,   # Transformed test data (no target)
                    self.data_transformation_config.preprocessor_obj_file_path,  # Path to saved preprocessor object
                )  

            except Exception as e:
                raise CustomException(e,sys)