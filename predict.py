import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

combined_df = pd.read_csv('cocoa_combined.csv')

# Standardize predictors
X = combined_df[['overall_tavg', 'US_Dollar_Index', 'Sugar_Price']]
X_standardized = (X - X.mean()) / X.std()
combined_df.describe()
# Define the target variable
Y = combined_df['Close']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
rf_predictions = rf.predict(X_test)
mse_rf = mean_squared_error(Y_test, rf_predictions)

# Gradient Boosting Machine
gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm.fit(X_train, Y_train)
gbm_predictions = gbm.predict(X_test)
mse_gbm = mean_squared_error(Y_test, gbm_predictions)

# LSTM Model Preparation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_df)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

look_back = 10
X_lstm, Y_lstm = create_dataset(scaled_data, look_back)
X_train_lstm, X_test_lstm, Y_train_lstm, Y_test_lstm = train_test_split(X_lstm, Y_lstm, test_size=0.2, random_state=42)

# LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(look_back, X_lstm.shape[2])))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, Y_train_lstm, epochs=20, batch_size=32, validation_data=(X_test_lstm, Y_test_lstm))

# Evaluate LSTM model
lstm_predictions = model_lstm.predict(X_test_lstm)
# Check and handle NaN values in LSTM predictions and Y_test_lstm
lstm_predictions_series = pd.Series(lstm_predictions.flatten())
Y_test_lstm_series = pd.Series(Y_test_lstm)

# Remove NaN values
lstm_predictions_clean = lstm_predictions_series.dropna()
Y_test_lstm_clean = Y_test_lstm_series.dropna()

# Ensure both arrays are the same length after cleaning
min_length = min(len(Y_test_lstm_clean), len(lstm_predictions_clean))
Y_test_lstm_clean = Y_test_lstm_clean.iloc[:min_length]
lstm_predictions_clean = lstm_predictions_clean.iloc[:min_length]

# Calculate mean squared error
mse_lstm = mean_squared_error(Y_test_lstm_clean, lstm_predictions_clean)

# Compile results into a table
results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'LSTM'],
    'MSE': [mse_rf, mse_gbm, mse_lstm]
}

results_df = pd.DataFrame(results)
print(results_df)