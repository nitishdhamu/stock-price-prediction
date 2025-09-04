#!/usr/bin/env python3
"""
app_streamlit.py — Stock Price Prediction (Quick below Mode + no empty box in Data Source)

Changes:
- Quick checkbox now sits directly under Mode toggle (center).
- Removed the empty uploader box that appeared in Online mode under Data Source.
"""

from __future__ import annotations
import os
import io
import hashlib
import calendar
from datetime import date
from typing import Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# core project functions (in stock_core.py)
from stock_core import fetch_data_yahoo_single, load_local_csv, run_pipeline

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
[data-testid="stHeader"] a, [data-testid="stMarkdownContainer"] a, [data-testid="stHeading"] a { display:none !important; }
.header-title { font-size:26px; font-weight:800; margin-bottom:6px; }
.header-sub { color:#94a3b8; margin-top:0; margin-bottom:10px; }
.downloads-col { display:flex; flex-direction:column; gap:8px; max-width:360px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("<div class='header-title'>📈 Stock Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>ARIMA, LSTM, GRU — choose Online (Yahoo) or upload a CSV (Offline).</div>", unsafe_allow_html=True)

# ---------------------------
# Mode + Quick
# ---------------------------
mode_label = st.radio("Mode", ["Online (Yahoo)", "Offline (CSV)"], index=0, horizontal=True)
quick_mode = st.checkbox("Quick mode (faster demo)", value=False, help="Quick = fewer epochs/lookback (faster but less accurate).")

mode_short = "Online" if mode_label.startswith("Online") else "Offline"

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Inputs")

    # Ticker
    st.markdown("### Ticker")
    ticker_choice = st.selectbox("Pick a ticker", ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "Custom"], index=0)
    ticker = st.text_input("Custom ticker", value="", placeholder="e.g., NFLX").strip().upper() if ticker_choice == "Custom" else ticker_choice

    st.markdown("---")

    # Duration
    st.markdown("### Duration")
    preset = st.selectbox("Range preset", ["Last N years", "YTD", "Max", "Custom"], index=0)
    if preset == "Last N years":
        years = st.number_input("Years back", min_value=1, max_value=30, value=5, step=1)
    elif preset == "YTD":
        st.caption("Year-to-date")
    elif preset == "Max":
        st.caption("Max available")
    else:
        st.caption("Custom range will be picked after clicking Run")

    st.markdown("---")

    # Data Source
    st.markdown("### Data Source")
    st.markdown(f"**Mode:** {mode_label}")
    if mode_short == "Offline":
        uploaded_csv = st.file_uploader("Upload CSV (Date & Adj Close)", type=["csv"])
        if uploaded_csv:
            st.markdown(f"- **Selected:** `{uploaded_csv.name}`")
    else:
        uploaded_csv = None
        st.caption("Online data fetched from Yahoo Finance.")

    st.markdown("---")

    # Advanced
    with st.expander("Advanced (optional)"):
        run_arima = st.checkbox("Run ARIMA", value=True)
        batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)

    RUN_ARIMA = run_arima if "run_arima" in locals() else True
    BATCH_SIZE = int(batch_size) if "batch_size" in locals() else 32

# ---------------------------
# Run button
# ---------------------------
run_clicked = st.button("Run", type="primary")

# ---------------------------
# Model parameters
# ---------------------------
LOOKBACK = 30 if quick_mode else 60
EPOCHS = 6 if quick_mode else 20
TEST_SIZE = 0.20
CSV_DATE_COL, CSV_PRICE_COL = "Date", "Adj Close"

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def csv_bytes_to_df(csv_bytes: bytes):
    buf = io.BytesIO(csv_bytes)
    return load_local_csv(buf, date_col=CSV_DATE_COL, price_col=CSV_PRICE_COL)

@st.cache_data(show_spinner=False)
def load_data_cached(mode_s, ticker_s, start, end, csv_bytes):
    tag = (ticker_s or "CSV").upper()
    if mode_s == "Online":
        df = fetch_data_yahoo_single(ticker_s, start, end)
    else:
        df = csv_bytes_to_df(csv_bytes)
    return df, tag

@st.cache_resource(show_spinner=False)
def train_cached(df, tag, lookback, test_size, epochs, batch_size, run_arima):
    return run_pipeline(df, tag, lookback, test_size, epochs, batch_size, run_arima)

def make_outdir(tag, start_s, end_s):
    out_dir = os.path.join("outputs", tag, f"{start_s}_to_{end_s}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_results(tag, start_s, end_s, metrics_df, preds_df, fig):
    out_dir = make_outdir(tag, start_s, end_s)
    mpath = os.path.join(out_dir, f"metrics_{tag}.csv")
    ppath = os.path.join(out_dir, f"predictions_{tag}.csv")
    fpath = os.path.join(out_dir, f"plot_{tag}.png")

    metrics_df.to_csv(mpath, index=False)
    preds_df.to_csv(ppath, index=False)
    fig.savefig(fpath, dpi=150)
    plt.close("all")

    st.session_state["metrics_csv"] = metrics_df.to_csv(index=False).encode("utf-8")
    st.session_state["preds_csv"] = preds_df.to_csv(index=False).encode("utf-8")
    st.session_state["plot_png"] = open(fpath, "rb").read()
    st.session_state["paths"] = {"metrics": mpath, "preds": ppath, "plot": fpath, "folder": out_dir}

def staged_run(df, tag):
    stages = ["Preparing", "Preprocessing", "Training LSTM", "Training GRU"]
    if RUN_ARIMA: stages.append("ARIMA")
    stages.append("Saving")
    prog, status = st.progress(0), st.empty()
    for i, s in enumerate(stages[:-1]):
        status.info(f"{s}...")
        prog.progress(int(i/len(stages)*100))
    metrics_df, preds_df, fig = train_cached(df, tag, LOOKBACK, TEST_SIZE, EPOCHS, BATCH_SIZE, RUN_ARIMA)
    prog.progress(100); status.success("Done"); prog.empty(); status.empty()
    return metrics_df, preds_df, fig

def render_results():
    p = st.session_state["paths"]
    st.subheader("📊 Metrics")
    st.dataframe(pd.read_csv(io.BytesIO(st.session_state["metrics_csv"])), use_container_width=True)
    st.subheader("📉 Actual vs Predicted")
    st.image(p["plot"], caption=os.path.basename(p["plot"]), use_container_width=True)
    st.subheader("⬇️ Downloads")
    st.markdown("<div class='downloads-col'>", unsafe_allow_html=True)
    st.download_button("Download Metrics CSV", st.session_state["metrics_csv"], file_name=os.path.basename(p["metrics"]))
    st.download_button("Download Predictions CSV", st.session_state["preds_csv"], file_name=os.path.basename(p["preds"]))
    st.download_button("Download Plot PNG", st.session_state["plot_png"], file_name=os.path.basename(p["plot"]))
    st.markdown("</div>", unsafe_allow_html=True)
    st.success(f"Saved to: `{p['folder']}`")

# ---------------------------
# Main run
# ---------------------------
if run_clicked:
    today = date.today()
    if preset == "Last N years":
        start_dt = date(today.year - int(years), today.month, today.day)
        start_str, end_str = start_dt.isoformat(), today.isoformat()
    elif preset == "YTD":
        start_str, end_str = date(today.year, 1, 1).isoformat(), today.isoformat()
    elif preset == "Max":
        start_str, end_str = "1900-01-01", today.isoformat()
    else:
        start_str, end_str = date(today.year-5, 1, 1).isoformat(), today.isoformat()

    csv_bytes = uploaded_csv.getvalue() if uploaded_csv else None
    if mode_short == "Offline" and not csv_bytes:
        st.error("Offline mode requires a CSV upload."); st.stop()
    if mode_short == "Online" and not ticker:
        st.error("Online mode requires a ticker."); st.stop()

    df, tag = load_data_cached(mode_short, ticker if mode_short=="Online" else None, start_str, end_str, csv_bytes)
    st.success(f"Data loaded — {df.index.min().date()} → {df.index.max().date()} (rows: {len(df)})")
    st.line_chart(df.rename(columns={"adj_close": "Adj Close"}))

    if mode_short == "Offline" and csv_bytes:
        st.subheader("CSV Preview"); st.dataframe(df.head(8), use_container_width=True)

    st.info("Training depends on your hardware (CPU/GPU/RAM).")
    metrics_df, preds_df, fig = staged_run(df, tag)
    save_results(tag, start_str, end_str, metrics_df, preds_df, fig)
    render_results()
else:
    if "paths" in st.session_state:
        st.info("Showing last results.")
        render_results()
    else:
        st.info("Configure inputs and click Run to start predictions.")
