from retry_requests import retry
import openmeteo_requests
import requests_cache
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
import os
import plotly.express as px


class WeatherUtils:
    def __init__(self):
        self.hourly_variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "snowfall",
            "weather_code",
            "pressure_msl",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "cloud_cover",
            "cloud_cover_low",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "sunshine_duration",
            "is_day",
            "shortwave_radiation_instant",
            "direct_normal_irradiance_instant",
            "diffuse_radiation_instant",
            "weather_code",
        ]
        self.historic_url = "https://archive-api.open-meteo.com/v1/archive"
        self.forecast_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        self.wmo_effect_mapping = {
            0: 0,  # Clear sky - no effect
            1: 1,  # Mainly clear - very little effect
            2: 2,  # Partly cloudy - slight effect
            3: 3,  # Overcast - moderate effect
            45: 4,
            48: 4,  # Fog & Depositing rime fog - moderate effect
            51: 3,
            53: 4,
            55: 5,  # Drizzle - increasing effect by intensity
            56: 4,
            57: 5,  # Freezing drizzle - moderate to high effect
            61: 5,
            63: 6,
            65: 7,  # Rain - increasing effect by intensity
            66: 6,
            67: 7,  # Freezing rain - high effect
            71: 5,
            73: 6,
            75: 7,  # Snow - increasing effect by intensity
            77: 6,  # Snow grains - high effect
            80: 5,
            81: 6,
            82: 7,  # Rain showers - increasing effect by intensity
            85: 6,
            86: 7,  # Snow showers - high effect
            95: 7,  # Thunderstorm - high effect
            96: 8,
            99: 9,  # Thunderstorm with hail - very high effect
        }
        self.holidays_list = [
            "2024-01-01",
            "2024-01-21",
            "2024-03-29",
            "2024-04-01",
            "2024-04-28",
            "2024-05-01",
            "2024-05-20",
            "2024-08-01",
            "2024-08-05",
            "2024-11-30",
            "2024-12-25",
            "2024-12-26",
        ]
        self.timezone = "America/Barbados"
        self.categorical_columns = ["weather_effect", "month", "day", "quarter", "hour", "is_day", "is_holiday"]
        self.NOCT = 45
        self.temperature_coefficient = 0.45
        self.daily_variables = None

    def extract_hourly_data_from_api_response(self, response, hourly_variables):
        if not hourly_variables:
            return None

        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        for i in range(len(hourly_variables)):
            hourly_variable = hourly_variables[i]
            hourly_variable_values = hourly.Variables(i).ValuesAsNumpy()
            hourly_data[hourly_variable] = hourly_variable_values

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        # hourly_dataframe["date"] = hourly_dataframe["date"] - pd.Timedelta(hours=7)

        return hourly_dataframe

    def extract_daily_data_from_api_response(self, response, daily_variables):
        daily = response.Daily()

        if not daily_variables:
            return None

        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s"),
                end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            )
        }

        for i in range(len(daily_variables)):
            daily_variable = daily_variables[i]
            daily_variable_values = daily.Variables(i).ValuesAsNumpy()
            daily_data[daily_variable] = daily_variable_values

        daily_dataframe = pd.DataFrame(data=daily_data)
        daily_dataframe = daily_dataframe.loc[daily_dataframe.index.repeat(24)].reset_index(drop=True)

        # Generate an hourly date range matching the daily repeated values
        hourly_dates = pd.date_range(
            start=daily_dataframe["date"].iloc[0],
            end=daily_dataframe["date"].iloc[-1] + pd.Timedelta(hours=23),
            freq="h",
            # timezone=timezone
        )
        daily_dataframe["date"] = hourly_dates
        # daily_dataframe.index = daily_dataframe.index.tz_localize(timezone)

        # Display the updated `daily_dataframe` with hourly entries
        daily_dataframe = daily_dataframe.drop(columns=["date"])

        return daily_dataframe

    def fetch_weather_data_from_api_based_on_url(self, url, latitude, longitude, start_date, end_date, hourly_variables, daily_variables, timezone):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        # url =
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_variables,
            "daily": daily_variables,
            "timezone": timezone,
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        print(response.Hourly())

        hourly_dataframe = self.extract_hourly_data_from_api_response(response, hourly_variables)
        daily_dataframe = self.extract_daily_data_from_api_response(response, daily_variables)

        if hourly_dataframe is None and daily_dataframe is None:
            return None
        elif hourly_dataframe is None:
            weather_data = daily_dataframe
        elif daily_dataframe is None:
            weather_data = hourly_dataframe
        else:
            weather_data = pd.concat([hourly_dataframe, daily_dataframe], axis=1)

        weather_data.index = pd.to_datetime(weather_data["date"], utc=True)
        weather_data = weather_data.drop(columns=["date"])
        weather_data.index = weather_data.index.tz_convert(timezone).tz_localize(None)
        weather_data = weather_data.dropna()

        return weather_data

    def fetch_historic_data_from_api(self, latitude, longitude, start_date, end_date):
        historic_url = self.historic_url
        hourly_variables = self.hourly_variables
        daily_variables = self.daily_variables
        timezone = self.timezone

        weather_historic = self.fetch_weather_data_from_api_based_on_url(
            historic_url, latitude, longitude, start_date, end_date, hourly_variables, daily_variables, timezone
        ).dropna()

        weather_historic.columns = [f"{col}_historic" for col in weather_historic.columns]

        return weather_historic

    def fetch_forecast_data_from_api(self, latitude, longitude, start_date, end_date):
        forecast_url = self.forecast_url
        hourly_variables = self.hourly_variables
        daily_variables = self.daily_variables
        timezone = self.timezone

        weather_forecast = self.fetch_weather_data_from_api_based_on_url(
            forecast_url, latitude, longitude, start_date, end_date, hourly_variables, daily_variables, timezone
        ).dropna()

        weather_forecast.columns = [f"{col}_forecast" for col in weather_forecast.columns]

        return weather_forecast


weather_processor = WeatherUtils()
# weather_historic = weather_processor.fetch_historic_data_from_api(13.1765314, -59.6168188, "2024-01-01", "2024-07-31")
# weather_forecast = weather_processor.fetch_forecast_data_from_api(13.1765314, -59.6168188, "2024-01-01", "2024-07-31")
