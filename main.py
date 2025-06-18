
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor
import datetime
import calendar
import tensorflow as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Limit GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load and prepare data
data = pd.read_csv(r"C:\Users\chawt\Desktop\earthquake detection\database.csv")
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Convert human-preferred time to UNIX time
timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d + ' ' + t, '%m/%d/%Y %H:%M:%S')
        ts_unix = calendar.timegm(ts.timetuple())
        timestamp.append(ts_unix)
    except Exception:
        timestamp.append(np.nan)

# Add timestamp and clean data
data['Timestamp'] = timestamp
final_data = data.dropna(subset=['Timestamp'])

# Ensure numeric columns
final_data['Latitude'] = pd.to_numeric(final_data['Latitude'], errors='coerce')
final_data['Longitude'] = pd.to_numeric(final_data['Longitude'], errors='coerce')
final_data['Magnitude'] = pd.to_numeric(final_data['Magnitude'], errors='coerce')
final_data['Depth'] = pd.to_numeric(final_data['Depth'], errors='coerce')
final_data = final_data.dropna(subset=['Latitude', 'Longitude', 'Magnitude', 'Depth'])

# Reset index after cleaning
final_data = final_data.reset_index(drop=True)

# Normalize Timestamp
timestamp_scaler = MinMaxScaler()
final_data['Timestamp'] = timestamp_scaler.fit_transform(final_data[['Timestamp']])

# Transform Latitude and Longitude to spherical coordinates
final_data['Lat_sin'] = np.sin(np.radians(final_data['Latitude']))
final_data['Lat_cos'] = np.cos(np.radians(final_data['Latitude']))
final_data['Lon_sin'] = np.sin(np.radians(final_data['Longitude']))
final_data['Lon_cos'] = np.cos(np.radians(final_data['Longitude']))

# Prepare features and targets
X = final_data[['Timestamp', 'Lat_sin', 'Lat_cos', 'Lon_sin', 'Lon_cos']].values
y_magnitude = final_data['Magnitude'].values
y_depth = final_data['Depth'].values

# Split data and store indices
train_idx, test_idx = train_test_split(range(len(final_data)), test_size=0.2, random_state=42)
X_train = X[train_idx]
X_test = X[test_idx]
y_magnitude_train = y_magnitude[train_idx]
y_magnitude_test = y_magnitude[test_idx]
y_depth_train = y_depth[train_idx]
y_depth_test = y_depth[test_idx]

# Scale features and targets
scaler_X = StandardScaler()
scaler_magnitude = StandardScaler()
scaler_depth = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_magnitude_train_scaled = scaler_magnitude.fit_transform(y_magnitude_train.reshape(-1, 1)).ravel()
y_magnitude_test_scaled = scaler_magnitude.transform(y_magnitude_test.reshape(-1, 1)).ravel()

y_depth_train_scaled = scaler_depth.fit_transform(y_depth_train.reshape(-1, 1)).ravel()
y_depth_test_scaled = scaler_depth.transform(y_depth_test.reshape(-1, 1)).ravel()

# Define model
input_dim = X_train_scaled.shape[1]

def create_model(neurons=128, activation='relu', optimizer='adam', loss='mse'):
    inputs = Input(shape=(input_dim,))
    x = Dense(neurons, activation=activation)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(neurons // 2, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(neurons // 4, activation=activation)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Magnitude model
magnitude_model = KerasRegressor(model=create_model, verbose=1, callbacks=[early_stopping, lr_scheduler])
magnitude_grid = GridSearchCV(estimator=magnitude_model, param_grid={
    'model__neurons': [128],
    'model__activation': ['relu'],
    'model__optimizer': ['adam'],
    'epochs': [100],
    'batch_size': [64]
}, n_jobs=1, cv=3, scoring='neg_mean_squared_error', verbose=2)
magnitude_grid_result = magnitude_grid.fit(X_train_scaled, y_magnitude_train_scaled, validation_split=0.2)

# Depth model
depth_model = KerasRegressor(model=create_model, verbose=1, callbacks=[early_stopping, lr_scheduler])
depth_grid = GridSearchCV(estimator=depth_model, param_grid={
    'model__neurons': [128],
    'model__activation': ['relu'],
    'model__optimizer': ['adam'],
    'epochs': [100],
    'batch_size': [64]
}, n_jobs=1, cv=3, scoring='neg_mean_squared_error', verbose=2)
depth_grid_result = depth_grid.fit(X_train_scaled, y_depth_train_scaled, validation_split=0.2)

# Evaluate models
magnitude_best = magnitude_grid_result.best_estimator_
depth_best = depth_grid_result.best_estimator_

y_magnitude_pred_scaled = magnitude_best.predict(X_test_scaled)
y_depth_pred_scaled = depth_best.predict(X_test_scaled)

magnitude_mse = mean_squared_error(y_magnitude_test_scaled, y_magnitude_pred_scaled)
magnitude_r2 = r2_score(y_magnitude_test_scaled, y_magnitude_pred_scaled)
depth_mse = mean_squared_error(y_depth_test_scaled, y_depth_pred_scaled)
depth_r2 = r2_score(y_depth_test_scaled, y_depth_pred_scaled)

# Inverse transform
y_magnitude_pred = scaler_magnitude.inverse_transform(y_magnitude_pred_scaled.reshape(-1, 1)).ravel()
y_depth_pred = scaler_depth.inverse_transform(y_depth_pred_scaled.reshape(-1, 1)).ravel()
magnitude_mse_original = mean_squared_error(y_magnitude_test, y_magnitude_pred)
depth_mse_original = mean_squared_error(y_depth_test, y_depth_pred)

# Print results
print(f"Magnitude - Best: {magnitude_grid_result.best_score_:.4f} using {magnitude_grid_result.best_params_}")
print(f"Magnitude - Test MSE (standardized): {magnitude_mse:.4f}, R²: {magnitude_r2:.4f}")
print(f"Magnitude - MSE (original): {magnitude_mse_original:.4f}")
print(f"Depth - Best: {depth_grid_result.best_score_:.4f} using {depth_grid_result.best_params_}")
print(f"Depth - Test MSE (standardized): {depth_mse:.4f}, R²: {depth_r2:.4f}")
print(f"Depth - MSE (original): {depth_mse_original:.4f}")

# Visualize prediction errors on maps
# Magnitude error map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Miller())
scatter = ax.scatter(
    final_data['Longitude'].iloc[test_idx], final_data['Latitude'].iloc[test_idx],
    c=(y_magnitude_test - y_magnitude_pred), cmap='hot_r', s=50, transform=ccrs.PlateCarree(),
    vmin=-2, vmax=2
)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()
plt.colorbar(scatter, label='Magnitude Error (Predicted - Actual)')
plt.title('Magnitude Prediction Errors')
plt.show()

# Depth error map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Miller())
scatter = ax.scatter(
    final_data['Longitude'].iloc[test_idx], final_data['Latitude'].iloc[test_idx],
    c=(y_depth_test - y_depth_pred), cmap='hot_r', s=50, transform=ccrs.PlateCarree(),
    vmin=-200, vmax=200
)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_global()
plt.colorbar(scatter, label='Depth Error (Predicted - Actual, km)')
plt.title('Depth Prediction Errors')
plt.show()
