# 📈 Stock Price Prediction (LSTM + Streamlit)

A simple web application built with **Streamlit** and **TensorFlow/Keras** that predicts future stock prices using an **LSTM (Long Short-Term Memory) neural network**.  
It fetches data directly from **Yahoo Finance** and provides interactive visualization of both historical and predicted prices.

---

## 🚀 Features

- Fetches stock data from **Yahoo Finance**
- Select custom history period or use maximum available data
- Adjust forecast horizon (1–30 days ahead)
- Interactive charts for historical + predicted prices
- Latest actual price and forecast displayed at the top
- Lightweight LSTM model with auto-regressive forecasting

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io) — web app framework
- [TensorFlow / Keras](https://www.tensorflow.org) — deep learning
- [scikit-learn](https://scikit-learn.org) — preprocessing
- [yfinance](https://pypi.org/project/yfinance/) — stock data
- [NumPy](https://numpy.org), [Pandas](https://pandas.pydata.org) — data handling

---

## 📂 Project Structure

```
.
├── app.py             # Main Streamlit application
├── model.py           # LSTM model builder and training utilities
├── utils.py           # Data utilities (fetch, scale, sequence creation)
├── prebuild.py        # Optional TensorFlow warm-up script
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Create a virtual environment:

   **Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **Mac/Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. (Optional) Warm up TensorFlow to reduce first-run latency:
   ```bash
   python prebuild.py
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open the app in your browser (default: [http://localhost:8501](http://localhost:8501)).

---

## 📊 Example

- Enter ticker: `AAPL` (Apple), `MSFT` (Microsoft), or `RELIANCE.NS` (Reliance India).
- Choose history period: e.g., `12` months.
- Select forecast horizon: e.g., `7` days.
- Click **Run** → View predictions on chart & table.

---

## 🔧 Configuration

You can adjust hyperparameters inside `app.py`:

```python
DEFAULT_WINDOW_SIZE = 60
EPOCHS = 5
UNITS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.2
```

To use a pre-trained model, place files in `models/initial/`:

- `model_architecture.json`
- `initial_weights.h5`

---

## ⚠️ Notes

- This app is for **educational/demo purposes** only — not financial advice.
- Stock market data is noisy; LSTM predictions may vary.
- For serious use: extend training, use early stopping, and validate thoroughly.

---

## 🤝 Contributing

Pull requests are welcome.  
For major changes, open an issue first to discuss what you’d like to change.

---

## 📄 License

This project is provided under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

---

## 👨‍💻 Author

Developed as part of an **AI/ML project**.  
For customization or client-specific deployment, please contact via GitHub profile.
