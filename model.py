"""
Model utilities: build and (optionally) train a simple LSTM model.
Kept minimal and documented for client handoff.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape, lr=0.001, units=50, dropout=0.2):
    """
    Build and compile a simple LSTM model.
    input_shape: (timesteps, features)
    """
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=32, verbose=1):
    """
    Train the model and return the history object.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return history
