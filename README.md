# 📈 Stock Price Prediction App  

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-brightgreen?style=for-the-badge)](https://stock-price-predictor2.streamlit.app/)  
*(Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit app link)*  

A machine learning-powered web application built with Python, scikit-learn, and Streamlit that predicts the next-day stock price and future movement (Up/Down) of a chosen stock.  

The app uses historical stock data (via yfinance), trains multiple regression and classification models, and provides backtesting metrics along with forecast results.  

---

## 🚀 Features

- Regression Models (Price Prediction)  
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  

- Classification Model (Movement Prediction)  
  - Logistic Regression predicts if the stock will go Up (1) or Down (0).  

- Metrics Provided  
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score  
  - Relative Error (% of Avg Price)  
  - Classification Accuracy & Confidence levels  

- Visualizations  
  - Actual vs Predicted (Regression)  
  - Classification (Up/Down movement with confidence)  

- Smart Forecasting  
  - Best regression model chosen by lowest MAE  
  - Average prediction across all models  
  - Forecast for next business day  

---


---

## ⚙️ How It Works

1. Data Collection  
   - Fetches stock data using Yahoo Finance (yfinance).  
   - User inputs ticker & training period.  

2. Feature Engineering  
   - Sliding window method: last N days (window_size) → predict next day.  

3. Model Training & Backtesting  
   - Walk-forward validation ensures real-world-like testing.  
   - Regression predicts price, classification predicts Up/Down.  

4. Forecasting  
   - Models retrained on full data.  
   - Predicts next trading day’s price & direction.  

---

## 🖥️ Usage

### 1. Clone Repository
```bash
git clone https://github.com/your-username/stock-predictor.git
cd stock-predictor

## 📷 Screenshots

### Home Page & Input Options
![Home Page](images/home.png)

### Prediction Results
![Results](images/results.png)

### Regression Models (Actual vs Predicted)
![Regression Chart](images/regression_chart.png)

### Classification (Up/Down Forecast)
![Classification Chart](images/classification_chart.png)


