
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 1. Generate synthetic time series data
np.random.seed(42)

time = np.arange(0, 200, 0.1)

data = np.sin(time) + 0.1 * np.random.randn(len(time))

plt.figure(figsize=(10, 4))
plt.plot(time, data)
plt.title("Synthetic Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# 2. Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 3. Create sequences
def create_sequences(dataset, seq_length):
    X, y = [], []

    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i + seq_length, 0])
        y.append(dataset[i + seq_length, 0])

    return np.array(X), np.array(y)

SEQ_LEN = 20

X, y = create_sequences(data_scaled, SEQ_LEN)

# 4. Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# 5. Train-test split
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build LSTM model
model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

# 7. Compile model
model.compile(
    optimizer='adam',
    loss='mse'
)

# 8. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 9. Plot training and validation loss
plt.figure(figsize=(8, 4))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 10. Forecast future values
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(
    predictions.reshape(-1, 1)
)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# 11. Plot actual vs predicted values
plt.figure(figsize=(10, 4))

plt.plot(y_test_actual, label='Actual')
plt.plot(predictions, label='Predicted')

plt.title("Actual vs Predicted Time Series")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.show()

print("Time Series Forecasting Completed Successfully!")
print("Number of predictions:", len(predictions))
