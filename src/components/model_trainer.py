import os
import sys
from dataclasses import dataclass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model_new.keras")

class ModelTRainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Split training and test input data")

            X_train = train_data
            X_test = test_data

            logging.info("Building the LSTM model")

             # Build the LSTM Autoencoder model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(64, return_sequences=False),  # Output only the last time step
                RepeatVector(X_train.shape[1]),  # Repeat the output across all time steps
                LSTM(64, return_sequences=True),
                LSTM(32, return_sequences=True),
                Dense(X_train.shape[2])  # Output layer with the same number of features as input
            ])

            # Compile the model
            model.compile(loss='mse', optimizer='adam')  # Mean Squared Error loss function
            logging.info("Model compiled successfully.")

            # Callbacks for early stopping, reducing learning rate, and saving the best model
            early_stopping = EarlyStopping(
                monitor='val_loss',  # Monitor validation loss
                patience=3,  # Number of epochs with no improvement before stopping
                restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',  # Reduce learning rate when validation loss stops improving
                factor=0.5,  # Reduce learning rate by this factor
                patience=3,  # Number of epochs with no improvement before reducing the learning rate
                min_lr=1e-6  # Lower bound on the learning rate
            )

            model_checkpoint = ModelCheckpoint(
                filepath=self.model_trainer_config.trained_model_file_path,  # File path to save the best model
                monitor='val_loss',  # Save model with the best validation loss
                save_best_only=True,  # Save only the best model
                save_weights_only=False,  # Save the full model (architecture + weights)
                verbose=1
            )

            # Train the model
            logging.info("Starting model training")
            history = model.fit(
                X_train, X_train,  # In anomaly detection, we train the model to reconstruct the input
                epochs=50,  # Number of epochs can be adjusted as needed
                batch_size=32,
                validation_data=(X_test, X_test),  # Validation on the test set
                callbacks = [early_stopping, reduce_lr, model_checkpoint]
            )

            logging.info("Model training completed.")

            # Save the trained model
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            return self.model_trainer_config.trained_model_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
