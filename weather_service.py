"""
Smart Shygyn PRO v3 — Weather Service
Real-time weather integration using Open-Meteo API (free, no API key required).
Includes caching to prevent API spam and fallback mechanisms.
"""

import streamlit as st
import requests
from typing import Tuple, Optional
from datetime import datetime


# City coordinates for weather API
CITY_COORDINATES = {
    "Алматы": (43.2389, 76.8897),
    "Астана": (51.1605, 71.4272),
    "Туркестан": (43.3031, 68.2717),
}


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_city_weather(city_name: str) -> Tuple[float, str, Optional[str]]:
    """
    Fetch current weather for a city using Open-Meteo API.
    
    API Documentation: https://open-meteo.com/en/docs
    
    Args:
        city_name: City name (must be in CITY_COORDINATES)
        
    Returns:
        Tuple of (temperature_celsius, weather_status, error_message)
        - temperature_celsius: Current temperature (float)
        - weather_status: "success" or "fallback"
        - error_message: Error description if fallback, None if success
    """
    # Validate city
    if city_name not in CITY_COORDINATES:
        return 15.0, "fallback", f"Unknown city: {city_name}"
    
    lat, lon = CITY_COORDINATES[city_name]
    
    # Open-Meteo API endpoint (free, no API key)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "timezone": "Asia/Almaty",
        "forecast_days": 1,
    }
    
    try:
        # Make request with timeout
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract current temperature
        if "current" in data and "temperature_2m" in data["current"]:
            temperature = float(data["current"]["temperature_2m"])
            return temperature, "success", None
        else:
            return 15.0, "fallback", "API response missing temperature data"
            
    except requests.exceptions.Timeout:
        return 15.0, "fallback", "API timeout (>5 seconds)"
    
    except requests.exceptions.ConnectionError:
        return 15.0, "fallback", "No internet connection"
    
    except requests.exceptions.HTTPError as e:
        return 15.0, "fallback", f"HTTP error: {e.response.status_code}"
    
    except Exception as e:
        return 15.0, "fallback", f"Unexpected error: {str(e)[:50]}"


def get_frost_multiplier(temperature: float) -> float:
    """
    Calculate frost heave failure probability multiplier.
    
    Physics model:
    - T >= 0°C: Normal risk (1.0×)
    - -5°C < T < 0°C: Moderate frost risk (1.1×)
    - T <= -5°C: Severe frost risk (1.2×)
    
    Args:
        temperature: Temperature in Celsius
        
    Returns:
        Multiplier for pipe failure probability (1.0 - 1.2)
    """
    if temperature >= 0:
        return 1.0
    elif temperature > -5:
        return 1.1
    else:
        return 1.2


def format_weather_display(city_name: str, 
                          temperature: float, 
                          status: str, 
                          error: Optional[str] = None) -> str:
    """
    Format weather information for display in Streamlit UI.
    
    Args:
        city_name: City name
        temperature: Temperature in Celsius
        status: "success" or "fallback"
        error: Error message if fallback
        
    Returns:
        Formatted HTML string for st.markdown
    """
    if status == "success":
        # Determine emoji based on temperature
        if temperature < -10:
            emoji = "🥶"
        elif temperature < 0:
            emoji = "❄️"
        elif temperature < 15:
            emoji = "🌤️"
        elif temperature < 25:
            emoji = "☀️"
        else:
            emoji = "🔥"
        
        return f"{emoji} **{temperature:.1f}°C** (Real-time)"
    else:
        return f"⚠️ **{temperature:.1f}°C** (Fallback: {error})"


def clear_weather_cache():
    """Clear the weather cache to force refresh."""
    get_city_weather.clear()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Weather Service Test")
    print("=" * 60)
    
    for city in CITY_COORDINATES.keys():
        temp, status, error = get_city_weather(city)
        frost_mult = get_frost_multiplier(temp)
        
        print(f"\n{city}:")
        print(f"  Temperature: {temp:.1f}°C")
        print(f"  Status: {status}")
        if error:
            print(f"  Error: {error}")
        print(f"  Frost Multiplier: {frost_mult}×")
    
    print("\n" + "=" * 60)
