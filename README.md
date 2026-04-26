# LSTM Volatility Prediction — URA Uranium ETF

> A deep learning approach to forecasting 21-day forward realised volatility on the **URA (Global X Uranium ETF)** using stacked LSTM recurrent neural networks — benchmarked against a classical **GARCH(1,1)** baseline.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-LSTM-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## Overview

Volatility is one of the hardest signals to forecast in financial markets — it clusters, jumps, and reacts non-linearly to regime shifts. This project tackles that challenge on the **URA uranium ETF** (an asset with a particularly volatile history) by training a stacked LSTM to learn temporal patterns in price-based features and predict realised volatility 21 trading days ahead.

The model is benchmarked head-to-head against **GARCH(1,1)**, the classical econometric workhorse, on the same out-of-sample window.

---

## Highlights

- **End-to-end Jupyter notebook** — from raw OHLCV download to model evaluation and report-ready plots.
- **Stacked LSTM** (64 → 32 units) with dropout regularisation and a custom loss function combining MSE with a variance-matching penalty.
- **8 engineered features**: log returns, rolling means/stds (5- and 21-day), volume change, intraday range, and open-close range.
- **GARCH(1,1) baseline** computed via the `arch` library for a fair classical-vs-deep-learning comparison.
- **Adaptive training** — EarlyStopping + ReduceLROnPlateau prevent overfitting on limited financial data.
- **Reproducible** — fixed random seeds and chronological train/val/test split (70/15/15).

---

## Results

Out-of-sample test set (Nov 2023 → Mar 2026, 516 sequences):

| Model            |     R²    |   RMSE   |    MAE   |
| ---------------- | :-------: | :------: | :------: |
| **LSTM (ours)**  | **0.033** | 0.00649  | 0.00506  |
| GARCH(1,1)       |  −0.158   | 0.00710  |   —      |

> The LSTM **outperforms GARCH(1,1) by ~0.19 in R²** on the test set, demonstrating that nonlinear temporal patterns provide meaningful signal beyond classical econometric models.

A full discussion — including residual analysis, training dynamics, and limitations — is in [`URA_Report_Final.pdf`](URA_Report_Final.pdf).

---

## Tech Stack

| Layer            | Tools                                     |
| ---------------- | ----------------------------------------- |
| Language         | Python 3.10+                              |
| Deep learning    | TensorFlow / Keras                        |
| Classical model  | `arch` (GARCH)                            |
| Data             | `yfinance`, `pandas`, `numpy`             |
| ML utilities     | `scikit-learn`                            |
| Visualisation    | `matplotlib`                              |
| Environment      | Jupyter Notebook                          |

---

## Architecture

```
Input (60 timesteps × 8 features)
   │
   ├─ LSTM(64, return_sequences=True)
   ├─ Dropout(0.2)
   ├─ LSTM(32)
   ├─ Dropout(0.2)
   ├─ Dense(16, relu)
   └─ Dense(1, linear) ──► 21-day forward realised volatility
```

**Total parameters:** 31,649  
**Loss:** MSE + 0.3 × variance-matching penalty  
**Optimizer:** Adam (lr = 1e-4)

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/GeorgeLazos/ura_volatility_prediction_LTSM.git
cd ura_volatility_prediction_LTSM
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

pip install numpy pandas matplotlib scikit-learn tensorflow yfinance arch jupyter
```

### 3. Run the notebook

```bash
jupyter notebook ura_volatility_prediction_LSTM.ipynb
```

The notebook downloads URA data live from Yahoo Finance, so an internet connection is required on the first run.

---

## Project Structure

```
.
├── ura_volatility_prediction_LSTM.ipynb   # Main notebook (data → model → eval)
├── URA_Report_Final.pdf                   # Full written report
├── LICENSE                                # MIT
├── .gitignore
└── README.md
```

---

## Methodology

1. **Data acquisition** — Daily OHLCV for URA from inception (2010-11-05), via `yfinance`.
2. **Feature engineering** — Log returns, rolling means/stds, volume change, intraday/open-close ranges.
3. **Target** — 21-day forward realised volatility (rolling std of log returns).
4. **Sequence construction** — 60-day sliding windows.
5. **Scaling** — `StandardScaler` fit on training set only (no leakage).
6. **Training** — Adam, MSE + variance-penalty loss, EarlyStopping (patience 20), ReduceLROnPlateau (factor 0.5, patience 5).
7. **Evaluation** — RMSE, MAE, R² on out-of-sample test set; head-to-head vs GARCH(1,1).

---

## License

Released under the [MIT License](LICENSE) — Copyright © 2026 Georgios Lazos.

---

## Author

**Georgios Lazos**  
MSc AI & Machine Learning — Deep Learning Final Project  
[GitHub @GeorgeLazos](https://github.com/GeorgeLazos)
