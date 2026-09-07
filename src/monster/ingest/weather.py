from __future__ import annotations

import requests

BASE = "https://api.weather.gov"


def _headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept": "application/geo+json"}


def point_metadata(lat: float, lon: float, user_agent: str) -> dict:
    response = requests.get(
        f"{BASE}/points/{lat},{lon}", headers=_headers(user_agent), timeout=20
    )
    response.raise_for_status()
    return response.json()


def hourly_forecast(lat: float, lon: float, user_agent: str) -> dict:
    meta = point_metadata(lat, lon, user_agent)
    url = meta["properties"]["forecastHourly"]
    response = requests.get(url, headers=_headers(user_agent), timeout=20)
    response.raise_for_status()
    return response.json()
