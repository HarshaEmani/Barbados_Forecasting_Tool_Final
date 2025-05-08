import pickle
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from supabase import create_client, Client
import os
from datetime import datetime
import lightgbm as lgb
import pandas as pd
import numpy as np
from DB_Manager import DatabaseManager

# from RLSCombiner import RLSCombiner
import json


class ScalerManager:
    def __init__(
        self,
        method: str = "standard",
        feeder_id: int = None,
        version: str = None,
        purpose: str = "features",
        load_type: str = "base",
        scenario: str = "24hr",
    ):
        self.method = method
        self.feeder_id = feeder_id
        self.purpose = purpose
        self.version = version or datetime.now().strftime("%Y%m%d")
        self.scaler = self._init_scaler()
        self.scenario = scenario
        self.load_type = "base" if purpose == "features" else load_type

    def _init_scaler(self):
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler method")

    def fit_transform(self, X):
        result = self.scaler.fit_transform(X)
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(result, index=X.index, columns=X.columns)
        self.save_to_db()
        return result

    def fit(self, X):
        self.scaler.fit(X)
        print(f"Scaler fitted with {self.method} method")

        self.save_to_db()

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X, index=X.index, columns=X.columns)
        return X

    def transform(self, X):
        result = self.scaler.transform(X)
        print(f"Scaler transformed with {self.method} method")
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, index=X.index, columns=X.columns)

        return result

    def inverse_transform(self, X):
        result = self.scaler.inverse_transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, index=X.index, columns=X.columns)
        return result

    def save_to_db(self):
        if self.feeder_id is None:
            raise ValueError("feeder_id must be set to save scaler")
        db = DatabaseManager()
        db.save_scaler(self.feeder_id, self.scaler, self.version, self.purpose, self.load_type, self.scenario)

    def load_from_db(self):
        if self.feeder_id is None:
            raise ValueError("feeder_id must be set to load scaler")
        db = DatabaseManager()
        db_scaler = db.load_scaler(self.feeder_id, self.version, self.purpose, self.load_type, self.scenario)

        if db_scaler is None:
            return None

        self.scaler = db_scaler
        return self
