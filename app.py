import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

start = '2014-12-31'
end = '2024-12-27'

st.title("Stock Trend Prediction")

# User input for stock ticker
user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(user_input, start=start, end=end)

# Describing data
st.subheader("Data from 2015 to November 2024")
st.write(df.describe())

# Moving Average (100 days)
st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df['Close'].rolling(100).mean()  # Fixed rolling mean calculation
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100-Day MA', color='orange')
plt.plot(df['Close'], label='Closing Price', color='blue')  # Correct column name
plt.title(f"Closing Price of {user_input} (2015- November 2024)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid()
st.pyplot(fig)

# Moving Average (100 and 200 days)
st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100-Day MA', color='red')
plt.plot(ma200, label='200-Day MA', color='green')
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title(f"Closing Price of {user_input} (2015- November 2024)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load pre-trained model
model = load_model('keras_model.h5')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

# Convert to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predict = model.predict(x_test)
scaler_factor = 1 / scaler.scale_[0]
y_predict = y_predict * scaler_factor
y_test = y_test * scaler_factor

numdigits = int(math.log10(self.target)) + 1
print(numdigits)
# Plotting Predictions vs Originalz
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
