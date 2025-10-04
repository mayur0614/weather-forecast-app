import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import requests
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Fetch NASA POWER daily data
# -------------------------------
def fetch_nasa_data(lat, lon, start="20100101", end="20241231"):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start,
        "end": end,
        "latitude": lat,
        "longitude": lon,
        "parameters": "T2M,RH2M,WS10M,PRECTOTCORR,PS,ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,WS2M",
        "community": "AG",
        "format": "JSON"
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    if "properties" not in data_json or "parameter" not in data_json["properties"]:
        return pd.DataFrame()

    data = data_json["properties"]["parameter"]
    df = pd.DataFrame({
        "date": list(data["T2M"].keys()),
        "Temperature (°C)": list(data["T2M"].values()),
        "Humidity (%)": list(data["RH2M"].values()),
        "Wind Speed (m/s)": list(data["WS10M"].values()),
        "Precipitation (mm)": list(data["PRECTOTCORR"].values()),
        "Pressure (Pa)": list(data["PS"].values()),
        "Solar Radiation (W/m²)": list(data["ALLSKY_SFC_SW_DWN"].values()),
        "Max Temperature (°C)": list(data["T2M_MAX"].values()),
        "Min Temperature (°C)": list(data["T2M_MIN"].values()),
        "Wind Speed 2m (m/s)": list(data["WS2M"].values())
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors='coerce')
    # Remove any rows with invalid dates
    df = df.dropna(subset=['date'])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -------------------------------
# Feature Engineering - Simplified and More Accurate
# -------------------------------
def add_temporal_features(df):
    """Add essential temporal features without overcomplicating"""
    df = df.copy()
    
    # Basic temporal features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Seasonal features (most important for weather)
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Temperature range (important for weather patterns)
    if 'Max Temperature (°C)' in df.columns and 'Min Temperature (°C)' in df.columns:
        df['temp_range'] = df['Max Temperature (°C)'] - df['Min Temperature (°C)']
    
    # Simple lag features (only 1-2 days to avoid overfitting)
    for col in ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Precipitation (mm)']:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
    
    # Simple rolling mean (3-day to capture short-term trends)
    for col in ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)']:
        if col in df.columns:
            df[f'{col}_3d_mean'] = df[col].rolling(window=3, min_periods=1).mean()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# -------------------------------
# Create sequences
# -------------------------------
def create_sequences(data, past_days=30, future_days=30):
    X, y = [], []
    for i in range(len(data) - past_days - future_days):
        X.append(data[i:i+past_days])
        y.append(data[i+past_days:i+past_days+future_days])
    return np.array(X), np.array(y)

# -------------------------------
# Train Accurate LSTM Model
# -------------------------------
def train_lstm(df):
    # Focus only on core weather parameters for accuracy
    core_features = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"]
    
    # Add minimal temporal features
    df_enhanced = add_temporal_features(df)
    
    # Select only essential features to avoid overfitting
    essential_features = core_features + ['day_of_year', 'sin_day_of_year', 'cos_day_of_year']
    
    # Filter to only include features that exist in the dataframe
    feature_cols = [col for col in essential_features if col in df_enhanced.columns]
    
    # Use MinMaxScaler for better weather prediction (keeps values in realistic ranges)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_enhanced[feature_cols])
    
    # Create sequences with shorter lookback for better accuracy
    X, y = create_sequences(scaled, past_days=14, future_days=7)  # Shorter sequences for accuracy

    # Simple train/val split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Simplified but effective LSTM architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(14, len(feature_cols))),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(7 * len(feature_cols))  # Predict 7 days instead of 30
    ])
    
    # Simpler optimizer
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Basic callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss')
    ]
    
    # Train with fewer epochs but better validation
    history = model.fit(
        X_train, y_train.reshape(y_train.shape[0], -1),
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val.reshape(y_val.shape[0], -1)),
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate metrics only for core weather features
    val_pred = model.predict(X_val)
    val_pred = val_pred.reshape(y_val.shape[0], -1)
    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    
    metrics = {}
    for i, col in enumerate(feature_cols):
        if i < val_pred.shape[1]:
            mae = mean_absolute_error(y_val_flat[:, i], val_pred[:, i])
            mse = mean_squared_error(y_val_flat[:, i], val_pred[:, i])
            r2 = r2_score(y_val_flat[:, i], val_pred[:, i])
            metrics[col] = {'MAE': mae, 'MSE': mse, 'R2': r2}
    
    # Save model and scaler
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    model.save("rain_lstm_model.h5")
    
    return model, scaler, feature_cols, metrics, history

# -------------------------------
# Predict next 7 days (more accurate)
# -------------------------------
def predict_next_7days(model, scaler, df, feature_cols):
    """Predict next 7 days with better accuracy"""
    try:
        # Add temporal features to recent data
        df_enhanced = add_temporal_features(df)
        
        # Scale the enhanced features
        scaled = scaler.transform(df_enhanced[feature_cols])
        
        # Get last 14 days for prediction (matching training)
        last_14 = scaled[-14:].reshape(1, 14, len(feature_cols))
        pred_scaled = model.predict(last_14)
        pred_scaled = pred_scaled.reshape(7, len(feature_cols))  # 7 days prediction
        
        # Inverse transform predictions
        preds = scaler.inverse_transform(pred_scaled)
        
        # Create prediction dataframe with robust date handling
        last_date = df["date"].iloc[-1]
        if pd.isna(last_date):
            # Fallback to current date if last date is NaT
            last_date = datetime.now()
        
        # Ensure we have a valid datetime
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.Timestamp(last_date)
        
        future_dates = pd.date_range(last_date + timedelta(days=1), 
                                     periods=7, freq="D")
        df_pred = pd.DataFrame(preds, columns=feature_cols)
        df_pred["Date"] = future_dates
        
        # Apply realistic constraints to predictions
        df_pred = apply_weather_constraints(df_pred)
        
        # Select only the main weather parameters for display
        main_features = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"]
        display_features = [col for col in main_features if col in df_pred.columns]
        
        return df_pred[["Date"] + display_features], df_pred
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return empty dataframe with proper structure
        empty_df = pd.DataFrame(columns=["Date", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"])
        return empty_df, empty_df

def apply_weather_constraints(df_pred):
    """Apply realistic weather constraints to predictions"""
    df = df_pred.copy()
    
    # Temperature constraints (realistic ranges)
    if 'Temperature (°C)' in df.columns:
        df['Temperature (°C)'] = np.clip(df['Temperature (°C)'], -50, 60)
    
    # Humidity constraints (0-100%)
    if 'Humidity (%)' in df.columns:
        df['Humidity (%)'] = np.clip(df['Humidity (%)'], 0, 100)
    
    # Wind speed constraints (0-50 m/s)
    if 'Wind Speed (m/s)' in df.columns:
        df['Wind Speed (m/s)'] = np.clip(df['Wind Speed (m/s)'], 0, 50)
    
    # Precipitation constraints (0-200 mm/day)
    if 'Precipitation (mm)' in df.columns:
        df['Precipitation (mm)'] = np.clip(df['Precipitation (mm)'], 0, 200)
    
    return df

# -------------------------------
# Predict next 30 days (using 7-day model iteratively)
# -------------------------------
def predict_next_30days(model, scaler, df, feature_cols):
    """Predict 30 days by iteratively using 7-day predictions"""
    try:
        all_predictions = []
        current_df = df.copy()
        
        # Predict 30 days in 7-day chunks
        for i in range(5):  # 5 * 7 = 35 days, we'll take first 30
            # Get 7-day prediction
            pred_7d, _ = predict_next_7days(model, scaler, current_df, feature_cols)
            
            # Check if prediction is valid
            if pred_7d.empty:
                print(f"Warning: Empty prediction at iteration {i}")
                break
            
            # Add predictions to our list
            all_predictions.append(pred_7d)
            
            # Update current_df with the last few days of predictions for next iteration
            if i < 4:  # Don't do this on the last iteration
                # Create a new row for the next iteration
                last_pred = pred_7d.iloc[-1].copy()
                last_pred['Date'] = pred_7d['Date'].iloc[-1]
                
                # Add to current_df for next prediction
                new_row = pd.DataFrame([last_pred])
                current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # Combine all predictions
        if all_predictions:
            df_30d = pd.concat(all_predictions, ignore_index=True)
            # Take only first 30 days
            df_30d = df_30d.head(30)
            return df_30d, df_30d
        else:
            # Return empty dataframe if no valid predictions
            empty_df = pd.DataFrame(columns=["Date", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"])
            return empty_df, empty_df
            
    except Exception as e:
        print(f"Error in 30-day prediction: {e}")
        # Return empty dataframe with proper structure
        empty_df = pd.DataFrame(columns=["Date", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"])
        return empty_df, empty_df

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌦️ RainCheck - Accurate Weather Forecast")
st.markdown("**Simplified LSTM model focused on core weather parameters for realistic predictions**")

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
past_days = st.sidebar.slider("Lookback Days", 15, 60, 30, help="Number of past days to consider for prediction")
future_days = st.sidebar.slider("Prediction Days", 7, 30, 30, help="Number of future days to predict")

# Main input
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=19.9975, help="Enter latitude coordinate")
with col2:
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=73.7898, help="Enter longitude coordinate")

# Training options
st.sidebar.header("Training Options")
use_enhanced_features = st.sidebar.checkbox("Use Enhanced Features", value=True, help="Include temporal and statistical features")
show_training_metrics = st.sidebar.checkbox("Show Training Metrics", value=True, help="Display model performance metrics")

if st.button("🚀 Train Accurate Model & Predict", type="primary"):
    with st.spinner("Fetching historical weather data..."):
        # Fetch more historical data for better training
        df = fetch_nasa_data(lat, lon, start="20100101", end="20241231")
    
    if df.empty:
        st.error("❌ No NASA data available for this location. Please check coordinates.")
    else:
        st.success(f"✅ Fetched {len(df)} days of historical data")
        
        with st.spinner("Training accurate LSTM model (this may take a few minutes)..."):
            # Train the accurate model
            model, scaler, feature_cols, metrics, history = train_lstm(df)
        
        st.success("✅ Model training completed!")
        
        # Display training metrics
        if show_training_metrics:
            st.subheader("📊 Model Performance Metrics")
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(metrics).T
            metrics_df = metrics_df.round(4)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average MAE", f"{metrics_df['MAE'].mean():.3f}")
            with col2:
                st.metric("Average R²", f"{metrics_df['R2'].mean():.3f}")
            with col3:
                st.metric("Features Used", len(feature_cols))
            
            # Show detailed metrics
            st.dataframe(metrics_df, use_container_width=True)
            
            # Plot training history
            if history:
                st.subheader("📈 Training History")
                hist_df = pd.DataFrame(history.history)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(hist_df[['loss', 'val_loss']])
                with col2:
                    st.line_chart(hist_df[['mae', 'val_mae']])

        # Fetch recent data for prediction
        with st.spinner("Fetching recent data for prediction..."):
            recent_df = fetch_nasa_data(lat, lon,
                                        start=(datetime.today()-timedelta(days=400)).strftime("%Y%m%d"),
                                        end=datetime.today().strftime("%Y%m%d"))
        
        if len(recent_df) < 30:
            st.error("❌ Not enough recent data for prediction")
        else:
            with st.spinner("Generating 30-day weather forecast..."):
                pred_df_display, pred_df_full = predict_next_30days(model, scaler, recent_df, feature_cols)
            
            st.success("✅ Weather forecast generated!")
            
            # Display predictions
            st.subheader("🌤️ Predicted Weather for Next 30 Days")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_temp = pred_df_display["Temperature (°C)"].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            with col2:
                avg_humidity = pred_df_display["Humidity (%)"].mean()
                st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
            with col3:
                total_precip = pred_df_display["Precipitation (mm)"].sum()
                st.metric("Total Precipitation", f"{total_precip:.1f}mm")
            with col4:
                max_wind = pred_df_display["Wind Speed (m/s)"].max()
                st.metric("Max Wind Speed", f"{max_wind:.1f}m/s")
            
            # Interactive data table
            st.dataframe(pred_df_display, use_container_width=True)
            
            # Enhanced visualizations
            st.subheader("📊 Weather Trends")
            
            # Temperature and Humidity
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(pred_df_display.set_index("Date")[["Temperature (°C)"]], height=300)
            with col2:
                st.line_chart(pred_df_display.set_index("Date")[["Humidity (%)"]], height=300)
            
            # Wind and Precipitation
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(pred_df_display.set_index("Date")[["Wind Speed (m/s)"]], height=300)
            with col2:
                st.bar_chart(pred_df_display.set_index("Date")[["Precipitation (mm)"]], height=300)
            
            # Combined plot
            st.subheader("🌡️ Complete Weather Overview")
            st.line_chart(pred_df_display.set_index("Date"), height=400)
            
            # Weather insights
            st.subheader("🔍 Weather Insights")
            
            # Temperature insights
            temp_trend = "increasing" if pred_df_display["Temperature (°C)"].iloc[-1] > pred_df_display["Temperature (°C)"].iloc[0] else "decreasing"
            st.write(f"• Temperature trend: {temp_trend} over the next 30 days")
            
            # Precipitation insights
            rainy_days = (pred_df_display["Precipitation (mm)"] > 1).sum()
            st.write(f"• Expected rainy days: {rainy_days} out of 30 days")
            
            # Wind insights
            windy_days = (pred_df_display["Wind Speed (m/s)"] > 5).sum()
            st.write(f"• Expected windy days: {windy_days} out of 30 days")
            
            # Download option
            csv = pred_df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"weather_forecast_{lat}_{lon}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
