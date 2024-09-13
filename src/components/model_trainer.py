import os
import sys
from dataclasses import dataclass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from kerastuner.tuners import BayesianOptimization

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model_new.keras")
    errors_file_path = os.path.join("artifacts", "errors.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def build_model(self, hp, input_shape):
        """Build the LSTM Autoencoder model using hyperparameters from Bayesian Optimization."""
        # Build the LSTM Autoencoder model
        model = Sequential([
            LSTM(units=hp.Int('units_1', min_value=32, max_value=256, step=32), return_sequences=True, input_shape=input_shape),
            LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=False),
            RepeatVector(input_shape[0]),  # Repeat the output across all time steps
            LSTM(units=hp.Int('units_3', min_value=32, max_value=128, step=32), return_sequences=True),
            LSTM(units=hp.Int('units_4', min_value=32, max_value=64, step=32), return_sequences=True),
            Dense(input_shape[1])  # Output layer with the same number of features as input
        ])
        
        # Compile the model with the selected optimizer
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
            loss='mse'
        )
        return model
    
    def bayesian_optimization_tuning(self, X_train, X_val):
        """Perform Bayesian Optimization for the LSTM Autoencoder model."""
        input_shape = (X_train.shape[1], X_train.shape[2])

        tuner = BayesianOptimization(
            lambda hp: self.build_model(hp, input_shape),  # The model building function
            objective='val_loss',  # Minimizing validation loss (MSE)
            max_trials=3,  # Number of hyperparameter settings to try
            directory='bayesian_opt_logs',  # Directory where the logs will be saved
            project_name='lstm_autoencoder'
        )

        # Early Stopping callback to stop training if no improvement after patience epochs
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        # Search for the best hyperparameters
        tuner.search(
            X_train, X_train,
            epochs=25,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            batch_size=32
        )

        # Get the best hyperparameters
        return tuner
    
    def get_callbacks(self):
        """Generate the list of callbacks used during model training."""
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=3,  # Number of epochs with no improvement before stopping
            restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
        )

        # Model checkpoint callback
        model_checkpoint = ModelCheckpoint(
            filepath=self.model_trainer_config.trained_model_file_path,  # File path to save the best model
            monitor='val_loss',  # Save model with the best validation loss
            save_best_only=True,  # Save only the best model
            save_weights_only=False,  # Save the full model (architecture + weights)
            verbose=1
        )

        return [early_stopping, model_checkpoint]
    
    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Split training and test input data")

            X_train = train_data
            X_test = test_data

            logging.info("Performing Bayesian Optimization")

            tuner = self.bayesian_optimization_tuning(X_train, X_test)

            # Retrieve the best model from tuning
            best_model = tuner.get_best_models(num_models=1)[0]

            # Compile the model
            best_model.compile(loss='mse', optimizer='adam')  # Mean Squared Error loss function
            logging.info("Model compiled successfully.")

            # Train the model
            logging.info("Starting model training")
            history = best_model.fit(
                X_train, X_train,  # In anomaly detection, we train the model to reconstruct the input
                epochs=50,  # Number of epochs can be adjusted as needed
                batch_size=32,
                validation_data=(X_test, X_test),  # Validation on the test set
                callbacks=self.get_callbacks()
            )

            logging.info("Model training completed.")

            # Save the trained model
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            # Make predictions
            logging.info("Generating predictions for training and test data")
            train_predictions = best_model.predict(X_train)
            test_predictions = best_model.predict(X_test)

            # Calculate reconstruction errors for training and test data
            train_reconstruction_error = np.mean(np.square(X_train - train_predictions), axis=(1, 2))
            test_reconstruction_error = np.mean(np.square(X_test - test_predictions), axis=(1, 2))

            # Save reconstruction errors and threshold
            threshold = np.percentile(train_reconstruction_error, 99)  # 99th percentile as threshold

            logging.info(f"Saving reconstruction errors and threshold to {self.model_trainer_config.errors_file_path}")

            # Save the threshold and reconstruction errors
            with open(self.model_trainer_config.errors_file_path, 'w') as f:
                f.write(f"Threshold: {threshold}\n")

            logging.info("Threshold saved.")

            return self.model_trainer_config.trained_model_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
