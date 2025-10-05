import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
import joblib
import requests

app = FastAPI(title="RainCheck Weather Forecast API")

# -------------------------------
# Load pre-trained model and scaler
# -------------------------------
model = load_model("rain_lstm_model.h5", compile=False)  # compile=False avoids deserialization errors
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# -------------------------------
# Helper functions
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
    df = df.dropna(subset=['date'])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_temporal_features(df):
    df = df.copy()
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    if 'Max Temperature (°C)' in df.columns and 'Min Temperature (°C)' in df.columns:
        df['temp_range'] = df['Max Temperature (°C)'] - df['Min Temperature (°C)']
    for col in ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Precipitation (mm)']:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
    for col in ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)']:
        if col in df.columns:
            df[f'{col}_3d_mean'] = df[col].rolling(window=3, min_periods=1).mean()
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def create_sequences(data, past_days=30, future_days=29):
    X, y = [], []
    for i in range(len(data) - past_days - future_days):
        X.append(data[i:i+past_days])
        y.append(data[i+past_days:i+past_days+future_days])
    return np.array(X), np.array(y)

def apply_weather_constraints(df_pred):
    df = df_pred.copy()
    if 'Temperature (°C)' in df.columns:
        df['Temperature (°C)'] = np.clip(df['Temperature (°C)'], -50, 60)
    if 'Humidity (%)' in df.columns:
        df['Humidity (%)'] = np.clip(df['Humidity (%)'], 0, 100)
    if 'Wind Speed (m/s)' in df.columns:
        df['Wind Speed (m/s)'] = np.clip(df['Wind Speed (m/s)'], 0, 50)
    if 'Precipitation (mm)' in df.columns:
        df['Precipitation (mm)'] = np.clip(df['Precipitation (mm)'], 0, 200)
    return df

def predict_next_7days(model, scaler, df, feature_cols):
    df_enhanced = add_temporal_features(df)
    scaled = scaler.transform(df_enhanced[feature_cols])
    last_30 = scaled[-30:].reshape(1, 30, len(feature_cols))
    pred_scaled = model.predict(last_30)
    pred_scaled = pred_scaled.reshape(29, len(feature_cols))
    # Take only first 7 days for 7-day prediction
    pred_scaled = pred_scaled[:7]
    preds = scaler.inverse_transform(pred_scaled)
    
    last_date = df["date"].iloc[-1]
    if pd.isna(last_date):
        last_date = pd.Timestamp(datetime.now())
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=7)
    df_pred = pd.DataFrame(preds, columns=feature_cols)
    df_pred["Date"] = future_dates
    df_pred = apply_weather_constraints(df_pred)
    
    main_features = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"]
    display_features = [col for col in main_features if col in df_pred.columns]
    return df_pred[["Date"] + display_features]

def predict_next_29days(model, scaler, df, feature_cols):
    """Predict 29 days directly without repetition"""
    try:
        df_enhanced = add_temporal_features(df)
        scaled = scaler.transform(df_enhanced[feature_cols])
        last_30 = scaled[-30:].reshape(1, 30, len(feature_cols))
        pred_scaled = model.predict(last_30)
        pred_scaled = pred_scaled.reshape(29, len(feature_cols))
        preds = scaler.inverse_transform(pred_scaled)
        
        last_date = df["date"].iloc[-1]
        if pd.isna(last_date):
            last_date = pd.Timestamp(datetime.now())
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=29)
        df_pred = pd.DataFrame(preds, columns=feature_cols)
        df_pred["Date"] = future_dates
        df_pred = apply_weather_constraints(df_pred)
        
        main_features = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"]
        display_features = [col for col in main_features if col in df_pred.columns]
        return df_pred[["Date"] + display_features]
    except Exception as e:
        print(f"Error in 29-day prediction: {e}")
        return pd.DataFrame(columns=["Date", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Precipitation (mm)"])

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/forecast/7days")
def forecast_7days(lat: float, lon: float):
    df = fetch_nasa_data(lat, lon,
                         start=(datetime.today()-timedelta(days=400)).strftime("%Y%m%d"),
                         end=datetime.today().strftime("%Y%m%d"))
    if df.empty or len(df) < 14:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")
    return predict_next_7days(model, scaler, df, feature_cols).to_dict(orient="records")

@app.get("/forecast/29days")
def forecast_29days(lat: float, lon: float):
    df = fetch_nasa_data(lat, lon,
                         start=(datetime.today()-timedelta(days=400)).strftime("%Y%m%d"),
                         end=datetime.today().strftime("%Y%m%d"))
    if df.empty or len(df) < 30:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")
    return predict_next_29days(model, scaler, df, feature_cols).to_dict(orient="records")
