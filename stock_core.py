from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# -------------- Utilities --------------
def parse_tickers(ticker_text: str | None) -> list[str]:
    """Split comma/space separated tickers -> unique list preserving order."""
    if not ticker_text:
        return []
    raw = ticker_text.replace(",", " ").split()
    seen = set()
    out = []
    for t in raw:
        t = t.strip()
        if t and t.upper() not in seen:
            seen.add(t.upper())
            out.append(t.upper())
    return out


def ensure_bday(df: pd.DataFrame) -> pd.DataFrame:
    """Business-day frequency with forward-fill."""
    df = df.asfreq("B")
    df["adj_close"] = df["adj_close"].ffill()
    return df


# -------------- Data loading --------------
def fetch_data_yahoo_single(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """
    Fetch a single ticker from Yahoo and return DF with one column 'adj_close'.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df is None or df.empty:
        raise ValueError(f"Yahoo returned no data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        # Try to pick the exact ticker's Adj Close if present
        try:
            sub = df["Adj Close"][ticker]
            series = pd.Series(sub, name="adj_close")
            out = series.to_frame()
        except Exception:
            if "Adj Close" in df.columns.get_level_values(0):
                sub = df["Adj Close"]
                if isinstance(sub, pd.DataFrame) and sub.shape[1] == 1:
                    series = sub.iloc[:, 0]
                    out = series.rename("adj_close").to_frame()
                else:
                    raise ValueError(
                        f"Yahoo returned multiple tickers; please pass a single ticker. Got columns: {list(df.columns)}"
                    )
            else:
                sub = df.xs("Close", axis=1, level=0)
                if isinstance(sub, pd.DataFrame) and sub.shape[1] == 1:
                    out = sub.iloc[:, 0].rename("adj_close").to_frame()
                else:
                    raise ValueError("Could not resolve a single price series from Yahoo MultiIndex data.")
    else:
        if "Adj Close" not in df.columns:
            if "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            else:
                raise ValueError("Yahoo data missing 'Adj Close'/'Close'.")
        out = df[["Adj Close"]].rename(columns={"Adj Close": "adj_close"})

    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)
    return ensure_bday(out)


def load_local_csv(csv_like, date_col: str = "Date", price_col: str = "Adj Close") -> pd.DataFrame:
    """Load local CSV into DF with 'adj_close'."""
    df = pd.read_csv(csv_like)
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in CSV columns: {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not found in CSV columns: {list(df.columns)}")
    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    df.rename(columns={price_col: "adj_close"}, inplace=True)
    return ensure_bday(df)


def get_dataset_prefer_online(
    ticker: str | None,
    start: str,
    end: str | None,
    csv_path_or_file: str | None,
    date_col: str,
    price_col: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prefer Yahoo for the given ticker; if it fails and a CSV is provided, use CSV.
    """
    tag = ticker.upper() if ticker else "CSV"
    if ticker:
        try:
            df = fetch_data_yahoo_single(ticker, start, end)
            return df, tag
        except Exception:
            if csv_path_or_file is None:
                raise
            df = load_local_csv(csv_path_or_file, date_col, price_col)
            return df, tag
    if csv_path_or_file is None:
        raise ValueError("Provide at least a ticker or a CSV.")
    df = load_local_csv(csv_path_or_file, date_col, price_col)
    return df, tag


# -------------- Models --------------
def train_test_split_series(series: pd.Series, test_size: float = 0.2):
    n = len(series)
    n_test = max(1, int(n * test_size))
    return series.iloc[:-n_test], series.iloc[-n_test:]


def fit_arima_grid(train: pd.Series, p_values=(0, 1, 2, 3), d_values=(0,), q_values=(0, 1, 2)):
    """
    Fit small ARIMA grid. For returns we usually only need d=0. Keep grid small for speed.
    """
    best_fit, best_aic = None, np.inf
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                    fit = model.fit(method_kwargs={"warn_convergence": False})
                    if fit.aic < best_aic:
                        best_fit, best_aic = fit, fit.aic
                except Exception:
                    continue
    if best_fit is None:
        raise RuntimeError("ARIMA failed for all (p,d,q) tested.")
    return best_fit


def forecast_arima(fit, steps: int) -> np.ndarray:
    return np.asarray(fit.forecast(steps=steps)).ravel()


def create_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y


def build_lstm(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_gru(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# -------------- Metrics + Plot --------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


def plot_predictions(dates, actual, preds_dict, title):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual")
    for name, yhat in preds_dict.items():
        plt.plot(dates, yhat, label=f"{name} Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    return fig


# -------------- Pipeline --------------
def run_pipeline(
    df: pd.DataFrame,
    tag: str,
    lookback: int = 60,
    test_size: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    run_arima: bool = True,
):
    """
    Pipeline that:
      - Runs ARIMA on returns (stationary) and re-integrates forecasts to price level.
      - Trains LSTM/GRU on price series (scaled, scaler fit on train only).
      - Aligns all predictions to the same test dates.
    Returns: (metrics_df, preds_df, fig)
    """
    series = df["adj_close"].astype(float)
    train, test = train_test_split_series(series, test_size)
    n_test = len(test)

    # -------- ARIMA on RETURNS ----------
    arima_price_pred = None
    if run_arima:
        # build returns series from train
        train_ret = train.pct_change().dropna()
        if len(train_ret) < 5:
            raise RuntimeError("Not enough train return points for ARIMA. Reduce lookback or test_size.")
        arima_fit = fit_arima_grid(train_ret, p_values=(0, 1, 2, 3), d_values=(0,), q_values=(0, 1, 2))
        ret_forecast = forecast_arima(arima_fit, steps=n_test)  # predicted returns for each test step
        # re-integrate to price level: start from last train price
        last_train_price = float(train.iloc[-1])
        # cumulative product: price_t = last_price * prod(1 + r_1..r_t)
        cum_returns = np.cumprod(1.0 + ret_forecast)
        arima_price_pred = (last_train_price * cum_returns).astype(float)

    # -------- LSTM / GRU on PRICES ----------
    # fit scaler only on train prices to avoid leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values.reshape(-1, 1))
    scaled_full = scaler.transform(series.values.reshape(-1, 1)).flatten()

    # sequences
    if len(train) <= lookback:
        raise RuntimeError(f"Not enough training data relative to lookback. len(train)={len(train)}, lookback={lookback}.")
    X_all, y_all = create_sequences(scaled_full, lookback)
    if n_test > X_all.shape[0]:
        raise RuntimeError("Test set is longer than possible sequence count; reduce test_size or lookback.")
    test_start_idx = X_all.shape[0] - n_test
    if test_start_idx <= 0:
        raise RuntimeError("Computed test_start_idx <= 0 -- not enough sequence data for training.")

    X_train, y_train = X_all[:test_start_idx], y_all[:test_start_idx]
    X_test, y_test = X_all[test_start_idx:], y_all[test_start_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm(input_shape)
    gru_model = build_gru(input_shape)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, min_delta=1e-5)

    lstm_model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    gru_model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0).reshape(-1, 1)
    gru_pred_scaled = gru_model.predict(X_test, verbose=0).reshape(-1, 1)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
    gru_pred = scaler.inverse_transform(gru_pred_scaled).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # canonical test dates (use test.index)
    test_dates = test.index

    # -------- Metrics ----------
    rows = []
    if run_arima:
        rows.append(("ARIMA", rmse(test.values.astype(float), arima_price_pred), mape(test.values.astype(float), arima_price_pred)))
    rows.append(("LSTM", rmse(y_test_actual, lstm_pred), mape(y_test_actual, lstm_pred)))
    rows.append(("GRU", rmse(y_test_actual, gru_pred), mape(y_test_actual, gru_pred)))
    metrics_df = pd.DataFrame(rows, columns=["model", "rmse", "mape(%)"])

    # -------- Predictions DF (unified on test_dates) ----------
    parts = []
    if run_arima:
        parts.append(pd.DataFrame({"date": test_dates, "model": "ARIMA", "actual": test.values.astype(float).ravel(), "predicted": arima_price_pred}))
    parts.append(pd.DataFrame({"date": test_dates, "model": "LSTM", "actual": y_test_actual, "predicted": lstm_pred}))
    parts.append(pd.DataFrame({"date": test_dates, "model": "GRU", "actual": y_test_actual, "predicted": gru_pred}))
    preds_df = pd.concat(parts, ignore_index=True)

    # -------- Plot ----------
    preds_pivot = preds_df.pivot(index="date", columns="model", values="predicted")
    actual_series = pd.Series(test.values.astype(float).ravel(), index=test_dates)

    preds_dict = {}
    if run_arima and "ARIMA" in preds_pivot.columns:
        preds_dict["ARIMA"] = preds_pivot["ARIMA"].values
    if "LSTM" in preds_pivot.columns:
        preds_dict["LSTM"] = preds_pivot["LSTM"].values
    if "GRU" in preds_pivot.columns:
        preds_dict["GRU"] = preds_pivot["GRU"].values

    fig = plot_predictions(test_dates, actual_series.values, preds_dict, f"{tag} - Test")

    return metrics_df, preds_df, fig


# -------------- Saving --------------
def save_outputs(tag: str, metrics_df: pd.DataFrame, preds_df: pd.DataFrame, fig, output_base: str = "outputs") -> dict:
    tag_s = tag.replace("/", "_")
    out_dir = os.path.join(output_base, tag_s)
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, f"metrics_{tag_s}.csv")
    ppath = os.path.join(out_dir, f"predictions_{tag_s}.csv")
    fpath = os.path.join(out_dir, f"plot_{tag_s}.png")
    metrics_df.to_csv(mpath, index=False)
    preds_df.to_csv(ppath, index=False)
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return {"metrics": mpath, "predictions": ppath, "plot": fpath, "folder": out_dir}


# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser(description="Stock Prediction: ARIMA vs LSTM vs GRU (online-first, CSV fallback)")
    ap.add_argument("--tickers", type=str, required=True, help="Single or multiple tickers (comma/space separated)")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--csv_path", type=str, default=None, help="CSV fallback if Yahoo fails (applies to all tickers)")
    ap.add_argument("--date_col", type=str, default="Date")
    ap.add_argument("--price_col", type=str, default="Adj Close")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_arima", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default="outputs")
    args = ap.parse_args()

    tickers = parse_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No valid tickers provided.")

    for t in tickers:
        print(f"\n=== Running pipeline for {t} ===")
        df, tag = get_dataset_prefer_online(
            ticker=t,
            start=args.start,
            end=args.end,
            csv_path_or_file=args.csv_path,
            date_col=args.date_col,
            price_col=args.price_col,
        )
        metrics_df, preds_df, fig = run_pipeline(
            df=df,
            tag=tag,
            lookback=args.lookback,
            test_size=args.test_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            run_arima=(args.no_arima == 0),
        )
        paths = save_outputs(tag, metrics_df, preds_df, fig, output_base=args.output_dir)
        print(metrics_df.to_string(index=False))
        print(f"[saved] {paths['metrics']}\n        {paths['predictions']}\n        {paths['plot']}")
        print(f"[folder] {paths['folder']}")


if __name__ == "__main__":
    main()
