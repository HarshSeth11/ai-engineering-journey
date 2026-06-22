import random

def get_weather(city : str) -> dict:
    """Simulated weather API"""
    weather_data = {
        "delhi": {"temp": 45, "condition": "Sunny", "humidity": 45},
        "mumbai": {"temp": 32, "condition": "Humid", "humidity": 85},
        "bangalore": {"temp": 26, "condition": "Cloudy", "humidity": 65},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return {"city": city, **weather_data[city_lower]}
    return {"city": city, "temp": random.randint(20,40), "condition": "Unknown", "humidity": 50}
