from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
from DB_Manager import DatabaseManager


class Scaler:
    def __init__(self, feeder_id: int, scenario_type: str, train_start_date: str, train_end_date: str, scaler_type: str = "standard"):
        self.feeder_id = feeder_id
        self.scenario_type = scenario_type
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

        print("Train start date:", train_start_date)
        print("Train end date:", train_end_date)
        self.scaler = scaler_type
        self.client = DatabaseManager()  # Assuming DatabaseManager is already implemented
        self.METADATA_SCHEMA = "metadata"

    def get_feeder_stats(self, feeder_id: int, scenario_type: str, train_start_date: str, train_end_date: str) -> pd.DataFrame:
        """
        Load feeder statistics from the database.
        """

        combined_data = self.client.fetch_data(feeder_id, train_start_date, train_end_date)
        combined_data.index = combined_data.index.tz_convert("UTC").tz_localize(None)

        feeder_stats = combined_data.describe().drop(columns=["Feeder_ID"]).T
        feeder_stats.reset_index(inplace=True)
        feeder_stats.columns = ["feature_name", "count", "mean", "std", "min", "25th_percentile", "50th_percentile", "75th_percentile", "max"]

        return feeder_stats

    def fit_transform(self, X: pd.DataFrame):
        """
        Fit scaler, save stats, and scale the data.
        """

        self.fit(X)
        result = self.transform(X)
        return pd.DataFrame(result, index=X.index, columns=X.columns)

    def fit(self, X: pd.DataFrame):
        """
        Fit scaler and save stats. Returns fitted scaler.
        """

        feeder_stats = self.get_feeder_stats(self.feeder_id, self.scenario_type, self.train_start_date, self.train_end_date)
        self.client.save_feeder_stats(self.feeder_id, self.scenario_type, self.train_start_date, self.train_end_date, feeder_stats)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Load stats from DB and scale the new data accordingly.
        """
        feeder_stats_df = self.client.load_feeder_stats(self.feeder_id, self.scenario_type, self.train_start_date, self.train_end_date)
        if feeder_stats_df.empty:
            raise ValueError("No feeder statistics found. Cannot transform.")

        scaled_df = X.copy()

        for feature in scaled_df.columns:
            if feature in feeder_stats_df.index:
                if self.scaler == "standard":
                    mean = feeder_stats_df.loc[feature, "mean"]
                    std = feeder_stats_df.loc[feature, "std"]
                    if std == 0:
                        std = 1e-8  # avoid division by zero
                    scaled_df[feature] = (scaled_df[feature] - mean) / std

        return scaled_df

    def inverse_transform(self, X: pd.DataFrame):
        """
        Inverse scale the features.
        Currently supports only standard scaler.
        """
        feeder_stats_df = self.client.load_feeder_stats(self.feeder_id, self.scenario_type, self.train_start_date, self.train_end_date)
        if feeder_stats_df.empty:
            raise ValueError("No feeder statistics found. Cannot inverse transform.")

        inverted_df = X.copy()
        for feature in inverted_df.columns:
            if feature in feeder_stats_df.index:
                if self.scaler == "standard":
                    # Inverse transform for standard scaler
                    mean = feeder_stats_df.loc[feature, "mean"]
                    std = feeder_stats_df.loc[feature, "std"]
                    inverted_df[feature] = (inverted_df[feature] * std) + mean

        return inverted_df
