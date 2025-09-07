"""
Streamlit app for Stock Price Prediction (LSTM).

Minimal, production-ready. Only necessary comments and no unused code.
"""

import os
from datetime import date, timedelta

import streamlit as st

# Project defaults / hyperparameters
DEFAULT_WINDOW_SIZE = 60
EPOCHS = 5
UNITS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.2
MODEL_SAVE_DIR = "models/initial"
MIN_ALLOWED_WINDOW = 3

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction")

# ---------------- Sidebar: user controls ----------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo)", value="AAPL", placeholder="AAPL, MSFT, RELIANCE.NS")
    months = st.number_input("History period (months)", min_value=1, max_value=120, value=12, step=1)
    use_max = st.checkbox("Use maximum available history", value=False)

    today = date.today()
    if use_max:
        default_start = date(1970, 1, 1)
    else:
        year = today.year
        month = today.month - months
        while month <= 0:
            month += 12
            year -= 1
        day = min(today.day, 28)
        default_start = date(year, month, day)
    default_end = today

    start = st.date_input("Start", value=default_start)
    end = st.date_input("End", value=default_end)

    forecast_days = st.number_input("Forecast days", min_value=1, max_value=30, value=7, step=1,
                                    help="How many days to forecast ahead")

    # Emphasized run button style
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #d32f2f;
            color: white;
            width: 100%;
            height: 44px;
            border-radius: 6px;
            font-weight: 600;
        }
        div.stButton > button:first-child:hover {
            background-color: #b71c1c;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    run_button = st.button("Run", use_container_width=True)
# -------------------------------------------------------

# Lazy-import helper
def _lazy():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import model_from_json
    return {"np": np, "pd": pd, "MinMaxScaler": MinMaxScaler, "model_from_json": model_from_json}


# Cached data fetch
@st.cache_data(show_spinner=False)
def fetch_data_cached(ticker: str, months: int, use_max_flag: bool, start_dt, end_dt):
    libs = _lazy()
    yf = __import__("yfinance")
    pd = libs["pd"]

    if start_dt is not None and end_dt is not None and start_dt != end_dt:
        start_str = pd.to_datetime(start_dt).date().isoformat()
        end_plus = pd.to_datetime(end_dt) + pd.Timedelta(days=1)
        end_str = end_plus.date().isoformat()
        df = yf.download(ticker, start=start_str, end=end_str, interval="1d", progress=False)
    else:
        if use_max_flag:
            df = yf.download(ticker, period="max", interval="1d", progress=False)
        else:
            df = yf.download(ticker, period=f"{int(months)}mo", interval="1d", progress=False)

    if df is None or df.empty:
        return pd.DataFrame(columns=["close"])

    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index = pd.to_datetime(df.index)
    return df


# Cached model loader/builder
@st.cache_resource(show_spinner=False)
def load_or_build_model_cached(save_dir: str, window_size: int, units: int, dropout: float, lr: float):
    libs = _lazy()
    model_from_json = libs["model_from_json"]

    arch_path = os.path.join(save_dir, "model_architecture.json")
    weights_path = os.path.join(save_dir, "initial_weights.h5")

    if os.path.exists(arch_path) and os.path.exists(weights_path):
        with open(arch_path, "r") as f:
            arch_json = f.read()
        model = model_from_json(arch_json)
        model.load_weights(weights_path)
        from tensorflow.keras.optimizers import Adam
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        return model

    from model import build_lstm_model
    model = build_lstm_model(input_shape=(window_size, 1), lr=lr, units=units, dropout=dropout)
    return model


# Prepare series: uses utils.scale_series
def prepare_series(series, scaler=None):
    from utils import scale_series
    scaled, scaler = scale_series(series, scaler)
    return scaled, scaler


def create_sequences(values, window_size):
    from utils import create_sequences as u_create_sequences
    return u_create_sequences(values, window_size)


# ---------------- Main pipeline ----------------
if run_button:
    if not ticker or not ticker.strip():
        st.error("Please enter a ticker (e.g., AAPL).")
    else:
        raw_ticker = ticker.strip().upper()
        status = st.empty()

        try:
            use_max_flag = bool(use_max)
            status.info("Loading data...")
            df = fetch_data_cached(raw_ticker, months, use_max_flag, start, end)

            if df.empty:
                st.warning("No historical data found for the ticker/period specified.")
                st.stop()

            status.info("Preparing series...")
            scaled_vals, scaler = prepare_series(df["close"])

            available = len(scaled_vals)
            max_window = max(available - 1, 0)
            if max_window < MIN_ALLOWED_WINDOW:
                st.error("Not enough historical data to run the model.")
                st.stop()

            window_size_used = min(DEFAULT_WINDOW_SIZE, max_window)
            if window_size_used != DEFAULT_WINDOW_SIZE:
                st.info(f"Using window = {window_size_used}")

            X, y = create_sequences(scaled_vals, window_size_used)
            if len(X) < 1:
                st.error("Not enough sequences created.")
                st.stop()

            status.info("Loading model...")
            model = load_or_build_model_cached(
                save_dir=MODEL_SAVE_DIR,
                window_size=window_size_used,
                units=UNITS,
                dropout=DROPOUT,
                lr=LEARNING_RATE,
            )

            status.info("Training model...")
            split = max(1, int(len(X) * 0.95))
            X_train, y_train = X[:split], y[:split]
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0)
            status.success("Training complete.")

            # Forecast generation (auto-regressive in scaled space)
            n_forecast = int(forecast_days)
            libs = _lazy()
            np = libs["np"]
            pd = libs["pd"]

            last_window = scaled_vals[-window_size_used:].reshape(1, window_size_used, 1).astype("float32")
            preds_scaled = []
            current = last_window.copy()
            for _ in range(n_forecast):
                pred_scaled = float(model.predict(current, verbose=0)[0][0])
                preds_scaled.append(pred_scaled)
                current = np.roll(current, -1, axis=1)
                current[0, -1, 0] = pred_scaled

            preds_scaled = np.array(preds_scaled).reshape(-1, 1)
            preds = scaler.inverse_transform(preds_scaled).flatten()

            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, n_forecast + 1)]
            future_index = pd.to_datetime(future_dates)

            # Prepare DataFrames for display
            df_forecast = pd.DataFrame({"predicted_close": preds}, index=future_index)

            # Avoid ambiguous Series.rename(...) call which can trigger 'str' object is not callable
            hist_series = df["close"].copy()
            hist_series.name = "historical_close"

            forecast_series = df_forecast["predicted_close"].copy()
            forecast_series.name = "forecast_close"

            combined = pd.concat([hist_series, forecast_series], axis=1).sort_index()

            # Top metrics: latest actual and last predicted price
            try:
                latest_actual_val = float(hist_series.iloc[-1])
            except Exception:
                latest_actual_val = 0.0

            try:
                last_predicted_val = float(forecast_series.iloc[-1])
            except Exception:
                last_predicted_val = 0.0

            pct_change = None
            if latest_actual_val != 0.0:
                pct_change = (last_predicted_val - latest_actual_val) / latest_actual_val * 100.0

            col1, col2 = st.columns(2)
            col1.metric(label="📌 Latest Actual Price", value=f"${latest_actual_val:,.2f}")
            delta_val = f"{pct_change:+.2f}%" if pct_change is not None else ""
            col2.metric(label=f"🔮 Predicted Price ({future_index[-1].date().isoformat()})",
                        value=f"${last_predicted_val:,.2f}", delta=delta_val)

            st.divider()

            # Chart and table
            st.subheader(f"📈 {raw_ticker} — Historical ({months} month{'s' if months > 1 else ''}) & {n_forecast}-Day Forecast")
            st.line_chart(combined)

            st.subheader(f"📊 {n_forecast}-Day Forecast Data")
            df_forecast_display = pd.DataFrame({
                "date": [d.date().isoformat() for d in future_index],
                "predicted_close": preds
            }).set_index("date")
            st.dataframe(df_forecast_display.style.format({"predicted_close": "{:.2f}"}))

            status.success("Done.")

        except Exception as exc:
            st.exception(exc)
