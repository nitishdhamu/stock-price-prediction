from __future__ import annotations
import os
import io
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# core project functions
from stock_core import fetch_data_yahoo_single, load_local_csv, run_pipeline

st.set_page_config(page_title="Stock Price Prediction", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
[data-testid="stHeader"] a, [data-testid="stMarkdownContainer"] a, [data-testid="stHeading"] a { display:none !important; }
.header-title { font-size:26px; font-weight:800; margin-bottom:6px; text-align:left; }
.header-sub { color:#64748b; margin-top:0; margin-bottom:10px; text-align:left; }
.downloads-col { display:flex; flex-direction:column; gap:12px; max-width:420px; }
.card { border-radius:10px; padding:12px; box-shadow: 0 1px 6px rgba(15,23,42,0.06); background:#ffffff; }
div.stButton > button:first-child {
    width: 220px !important;
    height: 3em;
    font-size: 1.1em;
    font-weight: bold;
    text-align: left;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("<div class='header-title'>📈 Stock Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>ARIMA (on returns), LSTM, GRU. Choose Online (Yahoo) or upload CSV.</div>", unsafe_allow_html=True)

# ---------------------------
# Sidebar - Inputs
# ---------------------------
with st.sidebar:
    st.header("Inputs")
    mode_label = st.radio("Mode", ["Online (Yahoo)", "Offline (CSV)"], index=0)
    mode_short = "Online" if mode_label.startswith("Online") else "Offline"

    st.markdown("---")
    st.subheader("Ticker / File")
    if mode_short == "Online":
        ticker_choice = st.selectbox("Choose ticker or custom", ["AAPL", "MSFT", "GOOG", "Custom"], index=0)
        ticker = st.text_input("Custom ticker (if selected)", value="", placeholder="e.g., NFLX").strip().upper() if ticker_choice == "Custom" else ticker_choice
        uploaded_csv = None
    else:
        uploaded_csv = st.file_uploader("Upload CSV (must contain Date and Adj Close columns)", type=["csv"])
        if uploaded_csv:
            st.markdown(f"- **Selected file:** `{uploaded_csv.name}`")
        ticker = None

    st.markdown("---")
    st.subheader("Date Range")
    preset = st.selectbox("Range preset", ["Last N years", "YTD", "Max", "Custom"], index=0)
    years = 5
    if preset == "Last N years":
        years = st.number_input("Years back", min_value=1, max_value=30, value=5, step=1)
    elif preset == "YTD":
        st.caption("Year-to-date (from Jan 1 this year)")
    elif preset == "Max":
        st.caption("Use max available history")
    else:
        st.date_input("Custom start date", value=date.today() - timedelta(days=365*5), key="custom_start")
        st.date_input("Custom end date", value=date.today(), key="custom_end")

    st.markdown("---")
    st.subheader("Model / Training")
    quick_mode = st.checkbox("Quick mode (faster demo)", value=False)
    st.caption("Quick mode reduces epochs & lookback for faster runs.")

    with st.expander("Advanced settings"):
        run_arima = st.checkbox("Run ARIMA", value=True)
        run_lstm = st.checkbox("Run LSTM", value=True)
        run_gru = st.checkbox("Run GRU", value=True)
        batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=20, step=1)
        lookback = st.number_input("Lookback (days)", min_value=1, max_value=365, value=60, step=1)

    st.markdown("---")
    st.caption("Outputs will be saved to `outputs/<TICKER>/<start>_to_<end>/`.")

# ---------------------------
# Run button
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)
run_clicked = st.button("🚀 Run", type="primary", key="run_main")
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Derived defaults
# ---------------------------
DEFAULT_LOOKBACK = 30 if quick_mode else 60
DEFAULT_EPOCHS = 6 if quick_mode else 20

LOOKBACK = int(lookback) if 'lookback' in locals() and lookback else DEFAULT_LOOKBACK
EPOCHS = int(epochs) if 'epochs' in locals() and epochs else DEFAULT_EPOCHS
TEST_SIZE = 0.20
CSV_DATE_COL, CSV_PRICE_COL = "Date", "Adj Close"

RUN_ARIMA = run_arima if 'run_arima' in locals() else True
RUN_LSTM = run_lstm if 'run_lstm' in locals() else True
RUN_GRU = run_gru if 'run_gru' in locals() else True
BATCH_SIZE = int(batch_size) if 'batch_size' in locals() else 32

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def csv_bytes_to_df(csv_bytes: bytes):
    buf = io.BytesIO(csv_bytes)
    return load_local_csv(buf, date_col=CSV_DATE_COL, price_col=CSV_PRICE_COL)

@st.cache_data(show_spinner=False)
def fetch_or_load(mode_s, ticker_s, start_s, end_s, csv_bytes):
    tag = (ticker_s or "CSV").upper()
    if mode_s == "Online":
        df = fetch_data_yahoo_single(ticker_s, start_s, end_s)
    else:
        df = csv_bytes_to_df(csv_bytes)
    return df, tag

def make_outdir(tag, start_s, end_s):
    safe_tag = tag.replace("/", "_").upper()
    out_dir = os.path.join("outputs", safe_tag, f"{start_s}_to_{end_s}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_results(tag, start_s, end_s, metrics_df, preds_df, fig):
    out_dir = make_outdir(tag, start_s, end_s)
    mpath = os.path.join(out_dir, f"metrics_{tag}.csv")
    ppath = os.path.join(out_dir, f"predictions_{tag}.csv")
    fpath = os.path.join(out_dir, f"plot_{tag}.png")

    metrics_df.to_csv(mpath, index=False)
    preds_df.to_csv(ppath, index=False)

    if fig is not None:
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.figure(figsize=(10,4))
        if "actual" in preds_df.columns and "predicted" in preds_df.columns:
            plt.plot(pd.to_datetime(preds_df["date"]), preds_df["actual"], label="Actual")
            plt.plot(pd.to_datetime(preds_df["date"]), preds_df["predicted"], label="Predicted")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fpath, dpi=150)
        plt.close("all")

    st.session_state["metrics_csv"] = open(mpath, "rb").read()
    st.session_state["preds_csv"] = open(ppath, "rb").read()
    st.session_state["plot_png"] = open(fpath, "rb").read()
    st.session_state["paths"] = {"metrics": mpath, "preds": ppath, "plot": fpath, "folder": out_dir}

def render_results():
    p = st.session_state["paths"]
    st.subheader("📊 Metrics")
    with st.expander("Metrics table (click to expand)"):
        metrics_df = pd.read_csv(io.BytesIO(st.session_state["metrics_csv"]))
        st.dataframe(metrics_df, width="stretch")

    st.subheader("📉 Actual vs Predicted")
    st.image(p["plot"], caption=os.path.basename(p["plot"]), width="stretch")

    st.subheader("⬇️ Downloads")
    st.markdown("<div class='downloads-col'>", unsafe_allow_html=True)
    st.download_button("Download Metrics CSV", st.session_state["metrics_csv"], file_name=os.path.basename(p["metrics"]))
    st.download_button("Download Predictions CSV", st.session_state["preds_csv"], file_name=os.path.basename(p["preds"]))
    st.download_button("Download Plot PNG", st.session_state["plot_png"], file_name=os.path.basename(p["plot"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.success(f"Saved to: `{p['folder']}`")

# ---------------------------
# Main run logic
# ---------------------------
if run_clicked:
    today = date.today()
    if preset == "Last N years":
        start_dt = date(max(1900, today.year - int(years)), today.month, today.day)
        start_str, end_str = start_dt.isoformat(), today.isoformat()
    elif preset == "YTD":
        start_str, end_str = date(today.year, 1, 1).isoformat(), today.isoformat()
    elif preset == "Max":
        start_str, end_str = "1900-01-01", today.isoformat()
    else:
        start_str = st.session_state.get("custom_start", (today - timedelta(days=365*5))).isoformat()
        end_str = st.session_state.get("custom_end", today).isoformat()

    if mode_short == "Offline" and not uploaded_csv:
        st.error("Offline mode requires you to upload a CSV file with 'Date' and 'Adj Close'.")
    elif mode_short == "Online" and (not ticker or ticker.strip() == ""):
        st.error("Online mode requires a ticker symbol.")
    else:
        csv_bytes = uploaded_csv.getvalue() if uploaded_csv else None
        with st.spinner("Loading data..."):
            df, tag = fetch_or_load(mode_short, ticker if mode_short == "Online" else None, start_str, end_str, csv_bytes)

        st.success(f"Data loaded — {df.index.min().date()} → {df.index.max().date()} (rows: {len(df)})")
        st.markdown("**Preview:**")
        st.dataframe(df.head(8), width="stretch")

        st.info("Training depends on your hardware (CPU/GPU/RAM). This may take some time.")
        prog = st.progress(0)
        status = st.empty()
        status.info("Preparing...")

        try:
            status.info("Preprocessing data...")
            prog.progress(10)

            status.info("Starting training...")
            prog.progress(30)

            metrics_df, preds_df, fig = run_pipeline(df, tag, LOOKBACK, TEST_SIZE, EPOCHS, BATCH_SIZE, RUN_ARIMA)

            prog.progress(90)
            status.info("Saving results...")
            save_results(tag, start_str, end_str, metrics_df, preds_df, fig)

            prog.progress(100)
            status.success("Done")
            render_results()
        except Exception as e:
            st.exception(f"Training failed: {e}")
        finally:
            prog.empty()
            status.empty()
else:
    if "paths" in st.session_state:
        st.info("Showing last results.")
        render_results()
    else:
        st.info("Configure inputs in the sidebar and press Run to start predictions.")
