# 🌦️ RainCheck Weather Forecast

An interactive weather forecasting application powered by LSTM neural networks, featuring both a REST API and a modern web interface.

## ✨ Features

- **🤖 AI-Powered Predictions**: Uses LSTM neural networks for accurate weather forecasting
- **📊 Interactive Dashboard**: Beautiful, responsive web interface with real-time charts
- **🔌 REST API**: Complete API for integration with other applications
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **📈 Advanced Visualizations**: Interactive charts using Plotly
- **🌍 Global Coverage**: Uses NASA POWER data for worldwide weather predictions
- **⚡ Real-time Updates**: Live data fetching and caching

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**:
   ```bash
   python start_app.py
   ```
   
   Or start components separately:
   
   **Start API server**:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```
   
   **Start web interface** (in another terminal):
   ```bash
   streamlit run app.py --server.port 8501
   ```

4. **Access the application**:
   - Web Interface: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - API Base URL: http://localhost:8000

## 📖 Usage

### Web Interface

1. **Open your browser** and go to http://localhost:8501
2. **Enter coordinates** for your desired location
3. **Select forecast period** (7 or 29 days)
4. **View interactive charts** and weather insights
5. **Download data** in CSV or JSON format

### API Usage

#### 7-Day Forecast
```bash
curl "http://localhost:8000/forecast/7days?lat=19.9975&lon=73.7898"
```

#### 29-Day Forecast
```bash
curl "http://localhost:8000/forecast/29days?lat=19.9975&lon=73.7898"
```

#### Python Example
```python
import requests

# Get 29-day forecast
response = requests.get(
    "http://localhost:8000/forecast/29days",
    params={"lat": 19.9975, "lon": 73.7898}
)

if response.status_code == 200:
    forecast_data = response.json()
    print(f"Forecast for {len(forecast_data)} days")
else:
    print(f"Error: {response.status_code}")
```

## 🏗️ Architecture

### Components

- **`api.py`**: FastAPI server with weather prediction endpoints
- **`app.py`**: Streamlit web interface with interactive dashboard
- **`start_app.py`**: Startup script for easy deployment
- **`requirements.txt`**: Python dependencies
- **`rain_lstm_model.h5`**: Pre-trained LSTM model
- **`scaler.pkl`**: Data scaler for model preprocessing
- **`feature_cols.pkl`**: Feature column definitions

### Data Flow

1. **Data Collection**: Fetches historical weather data from NASA POWER API
2. **Preprocessing**: Applies temporal features and scaling
3. **Prediction**: Uses trained LSTM model for forecasting
4. **Visualization**: Displays results in interactive charts
5. **Export**: Allows data download in multiple formats

## 🔧 Configuration

### API Configuration

Edit the `API_BASE_URL` in `app.py` to point to your API server:

```python
API_BASE_URL = "http://your-api-server:8000"
```

### Model Configuration

The model can be retrained with different parameters in `api.py`:

- **Lookback days**: 30 days of historical data
- **Prediction days**: 29 days forecast
- **Features**: Temperature, Humidity, Wind Speed, Precipitation

## 📊 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/forecast/7days` | GET | 7-day weather forecast | `lat`, `lon` |
| `/forecast/29days` | GET | 29-day weather forecast | `lat`, `lon` |
| `/docs` | GET | API documentation | - |

## 🛠️ Development

### Project Structure
```
raincheck-weather/
├── api.py              # FastAPI server
├── app.py              # Streamlit frontend
├── start_app.py        # Startup script
├── requirements.txt    # Dependencies
├── README.md          # This file
├── rain_lstm_model.h5 # Pre-trained model
├── scaler.pkl         # Data scaler
└── feature_cols.pkl   # Feature definitions
```

### Adding New Features

1. **API Endpoints**: Add new routes in `api.py`
2. **Frontend Components**: Add new UI elements in `app.py`
3. **Data Sources**: Modify data fetching functions
4. **Visualizations**: Add new chart types using Plotly

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Error**:
   - Ensure API server is running on port 8000
   - Check firewall settings
   - Verify API_BASE_URL in app.py

2. **Model Loading Error**:
   - Ensure model files exist in project directory
   - Check file permissions
   - Verify TensorFlow installation

3. **Data Fetching Error**:
   - Check internet connection
   - Verify NASA POWER API availability
   - Check coordinate validity

### Logs

- **API Logs**: Check terminal where uvicorn is running
- **Frontend Logs**: Check terminal where streamlit is running
- **Browser Console**: Check browser developer tools

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check browser console for frontend errors
4. Create an issue with detailed error information

---

**RainCheck Weather Forecast** - Powered by LSTM Neural Networks 🌦️
