import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def fetch_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance and return a DataFrame with datetime index.
    Keeps only the 'Close' column and renames it to 'close'.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["close"])
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index = pd.to_datetime(df.index)
    return df


def scale_series(series: pd.Series, scaler=None):
    """
    Scale series to [0,1]. Returns scaled array and scaler.
    """
    arr = series.values.reshape(-1, 1).astype("float32")
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)
    return scaled, scaler


def create_sequences(values: np.ndarray, window_size: int):
    """
    Create input sequences and targets for LSTM.
    values: 2D numpy array shape (n_samples, 1)
    Returns X (n, window_size, 1) and y (n, 1)
    """
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i: i + window_size])
        y.append(values[i + window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y
