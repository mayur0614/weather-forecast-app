import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

import os

# -------------------------------
# Configuration
# -------------------------------
IS_CLOUD_DEPLOYMENT = os.getenv("STREAMLIT_CLOUD_DEPLOYMENT", "false").lower() == "true"

# Set API URL based on environment
if IS_CLOUD_DEPLOYMENT:
    API_BASE_URL = None
    CLOUD_MODE = True
else:
    API_BASE_URL = "http://localhost:8000"
    CLOUD_MODE = False

st.set_page_config(
    page_title="🌦️ RainCheck Weather Forecast",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Geocoding Functions
# -------------------------------
@st.cache_data(ttl=3600)
def geocode_nominatim(location_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location_name, "format": "json", "limit": 1, "addressdetails": 1}
    headers = {"User-Agent": "RainCheck Weather App"}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data:
            location = data[0]
            return {'lat': float(location['lat']),
                    'lon': float(location['lon']),
                    'display_name': location['display_name'],
                    'address': location.get('address', {}),
                    'service': 'OpenStreetMap'}
    return None

def geocode_google(location_name, api_key):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location_name, "key": api_key}
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK' and data['results']:
            result = data['results'][0]
            location = result['geometry']['location']
            return {'lat': location['lat'],
                    'lon': location['lng'],
                    'display_name': result['formatted_address'],
                    'address': result.get('address_components', []),
                    'service': 'Google Maps'}
    return None

@st.cache_data(ttl=3600)
def geocode_location(location_name, api_key=None):
    if not location_name or location_name.strip() == "":
        return None, "Please enter a location name"
    
    services = [("OpenStreetMap", geocode_nominatim)]
    if api_key:
        services.append(("Google Maps", lambda loc: geocode_google(loc, api_key)))
    
    for service_name, service_func in services:
        try:
            result = service_func(location_name)
            if result and result.get('lat') and result.get('lon'):
                return result, None
        except Exception as e:
            st.warning(f"⚠️ {service_name} geocoding failed: {str(e)}")
            continue
    return None, "❌ Could not find location. Please try a different search term."

# -------------------------------
# Map
# -------------------------------
def create_location_map(lat, lon, location_name="Selected Location", zoom=10):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='OpenStreetMap')
    folium.Marker([lat, lon],
                  popup=f"<b>{location_name}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}",
                  tooltip=location_name,
                  icon=folium.Icon(color='red', icon='cloud', prefix='fa')).add_to(m)
    folium.Circle([lat, lon],
                  radius=5000,
                  popup="Weather forecast area",
                  color="blue",
                  fill=True,
                  fillColor="lightblue",
                  fillOpacity=0.2).add_to(m)
    return m

# -------------------------------
# Fallback Forecast
# -------------------------------
@st.cache_data(ttl=3600)
def generate_fallback_forecast(lat, lon, days=29):
    start_date = datetime.now() + timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    np.random.seed(int(lat * 1000 + lon))
    day_of_year = start_date.timetuple().tm_yday
    seasonal_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    base_temp = seasonal_temp + (lat - 40) * 0.5
    temperatures = base_temp + np.random.normal(0, 3, days)
    humidities = np.clip(60 + np.random.normal(0, 15, days), 20, 90)
    wind_speeds = np.clip(np.random.exponential(3, days), 0.5, 15)
    precip_prob = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25)
    precipitations = np.where(np.random.random(days) < precip_prob, np.random.exponential(2, days), 0)
    
    df = pd.DataFrame({
        'Date': dates,
        'Temperature (°C)': temperatures,
        'Humidity (%)': humidities,
        'Wind Speed (m/s)': wind_speeds,
        'Precipitation (mm)': precipitations
    })
    return df, None

# -------------------------------
# API Integration
# -------------------------------
@st.cache_data(ttl=300)
def fetch_weather_forecast(lat, lon, days=29):
    if CLOUD_MODE or API_BASE_URL is None:
        return generate_fallback_forecast(lat, lon, days)
    
    try:
        endpoint = f"{API_BASE_URL}/forecast/{days}days" if days != 7 else f"{API_BASE_URL}/forecast/7days"
        response = requests.get(endpoint, params={"lat": lat, "lon": lon}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df, None
        else:
            return pd.DataFrame(), f"API Error {response.status_code}: {response.text}"
    except:
        st.warning("⚠️ API unavailable, using fallback forecast")
        return generate_fallback_forecast(lat, lon, days)

def check_api_status():
    if CLOUD_MODE or API_BASE_URL is None:
        return True
    try:
        return requests.get(f"{API_BASE_URL}/docs", timeout=5).status_code == 200
    except:
        return False

# -------------------------------
# UI Components
# -------------------------------
def create_location_search_ui():
    st.subheader("📍 Location Search")
    search_method = st.radio("Choose search method:", ["🔍 Search by City/Address", "📍 Enter Coordinates Manually"], horizontal=True)
    
    if search_method == "🔍 Search by City/Address":
        col1, col2 = st.columns([3, 1])
        with col1:
            location_input = st.text_input("Enter city, address, or landmark:", placeholder="e.g., New York", help="Search for any location")
        with col2:
            google_api_key = st.text_input("Google Maps API Key (optional)", type="password")
        if st.button("🔍 Search Location", type="primary"):
            if location_input:
                with st.spinner("Searching..."):
                    result, error = geocode_location(location_input, google_api_key if google_api_key else None)
                if error:
                    st.error(error)
                else:
                    st.success(f"✅ Found: {result['display_name']}")
                    st.session_state['selected_location'] = result
                    st.session_state['lat'] = result['lat']
                    st.session_state['lon'] = result['lon']
                    st.session_state['location_name'] = result['display_name']
            else:
                st.warning("⚠️ Enter a location to search")
    
    else:  # Manual coordinates
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=19.9975, format="%.4f")
        with col2:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=73.7898, format="%.4f")
        if st.button("📍 Use These Coordinates", type="primary"):
            st.session_state['lat'] = lat
            st.session_state['lon'] = lon
            st.session_state['location_name'] = f"Custom Location ({lat:.4f}, {lon:.4f})"
            st.success(f"✅ Location set to {lat:.4f}, {lon:.4f}")
    
    if 'lat' in st.session_state and 'lon' in st.session_state:
        lat = st.session_state['lat']
        lon = st.session_state['lon']
        location_name = st.session_state.get('location_name', 'Selected Location')
        
        st.markdown("---")
        st.subheader("🗺️ Selected Location")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Location:** {location_name}")
            st.write(f"**Coordinates:** {lat:.4f}°N, {lon:.4f}°E")
            try:
                st_folium(create_location_map(lat, lon, location_name), width=600, height=400)
            except Exception as e:
                st.warning(f"⚠️ Could not display map: {str(e)}")
        with col2:
            st.markdown("**Location Details**")
            loc_data = st.session_state.get('selected_location', {})
            address = loc_data.get('address', {})
            if isinstance(address, dict):
                for key in ['city', 'state', 'country', 'postcode']:
                    if key in address:
                        st.write(f"**{key.title()}:** {address[key]}")
        return lat, lon, location_name
    
    return None, None, None

def create_weather_summary_cards(df):
    if df.empty:
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_temp = df["Temperature (°C)"].mean()
        temp_change = df['Temperature (°C)'].iloc[-1] - df['Temperature (°C)'].iloc[0]
        st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", f"{temp_change:+.1f}°C")
    with col2:
        avg_humidity = df["Humidity (%)"].mean()
        humidity_change = df['Humidity (%)'].iloc[-1] - df['Humidity (%)'].iloc[0]
        st.metric("💧 Avg Humidity", f"{avg_humidity:.1f}%", f"{humidity_change:+.1f}%")
    with col3:
        total_precip = df["Precipitation (mm)"].sum()
        max_precip = df['Precipitation (mm)'].max()
        st.metric("🌧️ Total Precipitation", f"{total_precip:.1f}mm", f"{max_precip:.1f}mm max")
    with col4:
        max_wind = df["Wind Speed (m/s)"].max()
        avg_wind = df['Wind Speed (m/s)'].mean()
        st.metric("💨 Max Wind Speed", f"{max_wind:.1f}m/s", f"{avg_wind:.1f}m/s avg")

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🌦️ RainCheck - Interactive Weather Forecast")
    if CLOUD_MODE:
        st.info("🌐 Cloud Mode: Using intelligent fallback forecasting")
    else:
        st.info("🏠 Local Mode: Real-time API-based forecast")
    
    if not check_api_status() and not CLOUD_MODE:
        st.error("⚠️ API server not running. Start with `uvicorn api:app --reload`")
        if st.button("🔄 Continue with Fallback Mode"):
            st.session_state['use_fallback'] = True
            st.rerun()
        else:
            st.stop()
    
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info("🌐 Cloud Mode" if CLOUD_MODE else "🏠 Local Mode with API")
    forecast_days = st.sidebar.selectbox("Forecast Period", [7, 29], index=1)
    show_charts = st.sidebar.checkbox("Show Interactive Charts", value=True)
    show_insights = st.sidebar.checkbox("Show Weather Insights", value=True)
    show_raw_data = st.sidebar.checkbox("Show Raw Data Table", value=False)
    
    st.markdown("---")
    lat, lon, location_name = create_location_search_ui()
    
    if lat is not None and lon is not None:
        st.markdown("---")
        st.subheader(f"🌤️ {forecast_days}-Day Weather Forecast for {location_name}")
        with st.spinner(f"Fetching forecast for {location_name}..."):
            df, error = fetch_weather_forecast(lat, lon, forecast_days)
        if error:
            st.error(error)
        else:
            create_weather_summary_cards(df)
            if show_charts:
                st.subheader("📈 Interactive Charts")
                # charts function can be added here
            if show_insights:
                st.subheader("🔍 Weather Insights & Recommendations")
                # insights function can be added here
            if show_raw_data:
                st.subheader("📋 Raw Forecast Data")
                st.dataframe(df)

if __name__ == "__main__":
    main()
