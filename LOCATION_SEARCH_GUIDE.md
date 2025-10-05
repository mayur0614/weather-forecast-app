# 🌍 Location Search Feature Guide

## Overview
The RainCheck Weather Forecast app now includes an advanced location search feature that allows users to search for weather forecasts by city name, address, or landmark instead of manually entering coordinates.

## ✨ Features

### 🔍 **Smart Location Search**
- **City Names**: Search for any city worldwide (e.g., "New York", "London", "Tokyo")
- **Addresses**: Search for specific addresses (e.g., "1600 Pennsylvania Avenue, Washington DC")
- **Landmarks**: Search for famous landmarks (e.g., "Eiffel Tower", "Times Square")
- **Country-Specific**: Include country names for better accuracy (e.g., "Paris, France")

### 🗺️ **Interactive Map**
- **Visual Location Display**: See your selected location on an interactive map
- **Location Details**: View detailed address information
- **Coordinate Display**: See exact latitude and longitude coordinates
- **Area Visualization**: Circle overlay showing the weather forecast area

### ⚙️ **Multiple Search Methods**
1. **Text Search**: Type any location name or address
2. **Manual Coordinates**: Enter latitude and longitude directly
3. **Google Maps Integration**: Optional Google Maps API for enhanced accuracy

## 🚀 How to Use

### **Step 1: Choose Search Method**
Select one of two options:
- **🔍 Search by City/Address**: Type any location name
- **📍 Enter Coordinates Manually**: Enter lat/lon directly

### **Step 2: Search for Location**
- **For Text Search**: Enter city, address, or landmark name
- **For Manual Entry**: Enter latitude (-90 to 90) and longitude (-180 to 180)

### **Step 3: View Results**
- **Location Confirmation**: See the found location details
- **Interactive Map**: View the location on a map
- **Coordinate Display**: See exact coordinates
- **Address Details**: View formatted address information

### **Step 4: Get Weather Forecast**
- **Automatic Forecast**: Weather forecast is automatically generated
- **Interactive Charts**: View temperature, humidity, wind, and precipitation trends
- **Weather Insights**: Get AI-powered recommendations

## 🔧 Configuration Options

### **Google Maps API (Optional)**
- **Purpose**: Enhanced geocoding accuracy and more detailed results
- **Setup**: Enter your Google Maps API key in the sidebar
- **Benefits**: More accurate location matching and detailed address information

### **API Key Setup**
1. Get a Google Maps API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Geocoding API
3. Enter the API key in the app's sidebar
4. Enjoy enhanced location search accuracy

## 🌐 Supported Location Types

### **Cities**
- Major cities: "New York", "London", "Tokyo", "Mumbai"
- With country: "Paris, France", "Sydney, Australia"
- With state: "Los Angeles, California", "Mumbai, Maharashtra"

### **Addresses**
- Street addresses: "123 Main Street, New York"
- Landmarks: "Times Square, New York"
- Government buildings: "White House, Washington DC"

### **International Locations**
- Any location worldwide
- Supports multiple languages
- Handles special characters and accents

## 📊 Location Data Structure

Each found location includes:
```json
{
  "lat": 40.7128,
  "lon": -74.0060,
  "display_name": "New York, NY, USA",
  "address": {
    "city": "New York",
    "state": "New York",
    "country": "United States"
  },
  "service": "OpenStreetMap"
}
```

## 🛠️ Technical Details

### **Geocoding Services**
1. **OpenStreetMap Nominatim** (Primary)
   - Free service
   - No API key required
   - Good global coverage
   - Rate-limited but sufficient for normal use

2. **Google Maps API** (Optional)
   - Enhanced accuracy
   - Requires API key
   - Higher rate limits
   - More detailed address information

### **Caching**
- Location searches are cached for 1 hour
- Reduces API calls and improves performance
- Cache is cleared when the app restarts

### **Error Handling**
- Comprehensive error messages
- Fallback to alternative geocoding services
- Clear troubleshooting guidance
- Graceful degradation when services are unavailable

## 🎯 Best Practices

### **For Better Search Results**
1. **Be Specific**: Include city and country when possible
2. **Use Common Names**: Use widely recognized location names
3. **Check Spelling**: Ensure correct spelling of location names
4. **Include Context**: Add state, province, or country for clarity

### **Examples of Good Searches**
- ✅ "New York, NY"
- ✅ "London, UK"
- ✅ "Tokyo, Japan"
- ✅ "Mumbai, Maharashtra, India"
- ✅ "Eiffel Tower, Paris"
- ✅ "Times Square, New York"

### **Examples of Less Effective Searches**
- ❌ "NY" (too vague)
- ❌ "The big city" (not specific)
- ❌ "My house" (personal reference)

## 🔍 Troubleshooting

### **Common Issues**

1. **"Location not found"**
   - Try a more specific search term
   - Include country or state name
   - Check spelling
   - Try alternative location names

2. **"Could not find location"**
   - Verify internet connection
   - Try a different search term
   - Check if the location exists
   - Use manual coordinates as fallback

3. **Map not displaying**
   - Check if folium is installed
   - Verify internet connection
   - Try refreshing the page

### **Fallback Options**
- Use manual coordinate entry
- Try different search terms
- Use Google Maps API if available
- Contact support for assistance

## 📱 Mobile Compatibility

The location search feature is fully responsive and works on:
- **Desktop**: Full interactive map and detailed interface
- **Tablet**: Optimized layout with touch-friendly controls
- **Mobile**: Streamlined interface with essential features

## 🔒 Privacy & Security

- **No Data Storage**: Location searches are not permanently stored
- **Temporary Caching**: Only cached for 1 hour for performance
- **API Key Security**: Google Maps API keys are handled securely
- **No Tracking**: No user location tracking or data collection

## 🚀 Future Enhancements

Planned improvements include:
- **Recent Locations**: Save frequently searched locations
- **Favorites**: Bookmark favorite locations
- **Location History**: Track search history
- **Advanced Filters**: Filter by country, region, or type
- **Voice Search**: Voice-activated location search
- **GPS Integration**: Use device location automatically

---

**RainCheck Weather Forecast** - Making weather prediction accessible worldwide! 🌦️
