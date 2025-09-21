# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# ================================
# Load Data & Models
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    return df

df = load_data()

# Features & Target
features = ['Close_Lag1', 'Close_Lag2', 'MA5', 'MA10']
X = df[features].values
y = df['Target'].values

# Load Scaler
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Train-test split (same as in main.ipynb)
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Load Ensemble Models
voting_model = joblib.load("voting_model.pkl")
stacking_model = joblib.load("stacking_model.pkl")

# Load LSTM Model
lstm_model = tf.keras.models.load_model("lstm_model.h5")

# Prepare LSTM input
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Predictions
y_pred_voting = voting_model.predict(X_test)
y_pred_stacking = stacking_model.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# ================================
# Evaluation Function
# ================================
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

rmse_v, mae_v, r2_v = evaluate_model(y_test, y_pred_voting)
rmse_s, mae_s, r2_s = evaluate_model(y_test, y_pred_stacking)
rmse_l, mae_l, r2_l = evaluate_model(y_test, y_pred_lstm)

# ================================
# Streamlit UI
# ================================
st.title("📈 Apple Stock Price Prediction")
st.markdown("""
This app loads **pretrained models** (Voting, Stacking, and LSTM)  
to predict Apple stock prices.  
The models were trained earlier in **main.ipynb**.
""")

# Stock Price History
st.subheader("📊 Stock Closing Price History")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['Date'], df['Close'], label="Close Price", color="blue")
ax.set_title("Apple Stock Closing Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# Show evaluation metrics
st.subheader("🔎 Model Performance Comparison")
results = pd.DataFrame({
    "Model": ["Voting Regressor", "Stacking Regressor", "LSTM"],
    "RMSE": [rmse_v, rmse_s, rmse_l],
    "MAE": [mae_v, mae_s, mae_l],
    "R²": [r2_v, r2_s, r2_l]
})
st.dataframe(results)

# Predictions vs Actual
st.subheader("📉 Predictions vs Actual")
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(df['Date'].iloc[-len(y_test):], y_test, label="Actual", color='blue')
ax2.plot(df['Date'].iloc[-len(y_test):], y_pred_voting, label="Voting", color='green', alpha=0.7)
ax2.plot(df['Date'].iloc[-len(y_test):], y_pred_stacking, label="Stacking", color='red', alpha=0.7)
ax2.plot(df['Date'].iloc[-len(y_test):], y_pred_lstm, label="LSTM", color='orange', alpha=0.7)
ax2.legend()
st.pyplot(fig2)

# Conclusion
st.subheader("✅ Conclusion")
st.success("Among all models, **LSTM performs the best** because it captures sequential patterns in stock data.")
