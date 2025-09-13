import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Tesla.csv')
scaler = MinMaxScaler()
open_price = df['Open'].values.reshape(-1, 1)
scaled_open_price = scaler.fit_transform(open_price)
train_size = int(len(scaled_open_price) * 0.75)
train_data = scaled_open_price[:train_size]
test_data = scaled_open_price[train_size:]

def create_dataset(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)
X_train, y_train = create_dataset(train_data)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])
model.save('tesla_stock_model.keras')
from tensorflow.keras.models import load_model
loaded_model = load_model('tesla_stock_model.keras')
