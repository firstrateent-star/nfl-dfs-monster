from __future__ import annotations
import requests

BASE = "https://api.weather.gov"

class NWSClient:
    def __init__(self, user_agent: str, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/geo+json"})
        self.timeout = timeout

    def hourly_forecast(self, latitude: float, longitude: float) -> dict:
        point = self.session.get(f"{BASE}/points/{latitude},{longitude}", timeout=self.timeout)
        point.raise_for_status()
        url = point.json()["properties"]["forecastHourly"]
        forecast = self.session.get(url, timeout=self.timeout)
        forecast.raise_for_status()
        return forecast.json()
