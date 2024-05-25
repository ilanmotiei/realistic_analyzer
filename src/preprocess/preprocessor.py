import pandas as pd
from tqdm import tqdm

from src.config import Config


class Preprocessor:
    @staticmethod
    def load_data() -> pd.DataFrame:
        return pd.read_csv(Config.DATA_LOCATION_FILEPATH)

    @staticmethod
    def add_geolocation(data: pd.DataFrame) -> pd.DataFrame:
        for i, row in data.iterrows():
            data.iloc[i, '']