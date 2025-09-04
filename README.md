# 📈 Stock Price Prediction (ML/Time Series)

⚠️ **For research & education only — not for financial or trading use.**

This project provides a complete machine learning pipeline for **stock price prediction** using **ARIMA, LSTM, and GRU** models.  
It supports fetching stock data from **Yahoo Finance (online)** or uploading your own **CSV (offline)**.  

You can use:  
- 🖥️ **Streamlit Web UI** (recommended)  
- 💻 **CLI (optional)** for automation  

---

## 🚀 Features

- Fetch stock data from **Yahoo Finance** or upload **CSV**  
- Train and compare **ARIMA, LSTM, GRU** models  
- Evaluate predictions with metrics: **RMSE** and **MAPE**  
- Visualize **actual vs predicted prices**  
- Save outputs into structured folders:
  ```
  outputs/<TICKER>/<START>_to_<END>/
  ├── metrics_<TICKER>.csv
  ├── predictions_<TICKER>.csv
  └── plot_<TICKER>.png
  ```

---

## 📥 Clone the Repository

```bash
git clone https://github.com/nitishdhamu/stock-price-prediction.git
cd stock-price-prediction
```

---

## ⚙️ Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Run Streamlit UI (Recommended)

```bash
streamlit run app_streamlit.py
```

- Select **Mode**: Online (Yahoo) or Offline (CSV)  
- Pick a ticker (e.g., `AAPL`, `MSFT`, etc.) or upload your own CSV  
- Choose duration, run models, and download results  

---

## 💻 Run CLI (Optional)

Run the pipeline directly from the command line:

```bash
python stock_core.py --ticker AAPL --years 5
```

Results will be saved to:

```
outputs/AAPL/<START>_to_<END>/
```

---

## 📊 Example Output

```
outputs/AAPL/2020-01-01_to_2025-01-01/
  ├── metrics_AAPL.csv
  ├── predictions_AAPL.csv
  └── plot_AAPL.png
```

---

## 📂 Project Structure

```
├── app_streamlit.py        # Streamlit UI
├── stock_core.py           # Core pipeline (data, training, saving)
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── LICENSE                 # MIT License
└── outputs/                # Results (per run)
```

---

## ⚖️ License

This project is licensed under the **MIT License**.  

---

## 🙋 About

- Built for **AI/ML Time Series analysis** learning, internships, and forecasting experiments  
- Includes both **UI for beginners** and **CLI (optional)** for advanced users  
