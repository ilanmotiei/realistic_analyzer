from typing import Dict, Any, Tuple, Union, Optional, List
import logging
import json
import os

import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely import Point
from datetime import datetime

from src.config import Config
from src.preprocess.utils import itm_to_wsg84
from src.preprocess.tokenizer import Tokenizer


logging.basicConfig(level=logging.INFO)


class Preprocessor:
    def __init__(self, radius_meters: float, settlements_subset: Optional[List[str]] = None) -> None:
        self.radius_meters = radius_meters
        self.settlements_subset = settlements_subset
        self.deals_gdf = None
        self.address_groups = None
        self.addresses_gdf = None
        self.addresses_edges = None

        self.tokenizer: Tokenizer = Tokenizer()
        self.init_data()
        self.init_graph()

    def init_data(self) -> None:
        deals_df: pd.DataFrame = pd.read_csv(Config.DATA_LOCATION_FILEPATH)
        deals_df: pd.DataFrame = deals_df.replace('nan', np.nan)
        if self.settlements_subset:
            deals_df: pd.DataFrame = deals_df[deals_df[Config.SETTLEMENT_FIELD_NAME].isin(self.settlements_subset)]

        deals_df: pd.DataFrame = Preprocessor.remove_unnecessary_columns(deals_df)
        deals_df: pd.DataFrame = Preprocessor.remove_irrelevant_deals(deals_df)
        deals_df: pd.DataFrame = Preprocessor.remove_hidden_characters(deals_df)
        deals_df: pd.DataFrame = Preprocessor.add_numeric_floor_numbers(deals_df)
        deals_df: pd.DataFrame = self.add_settlements_indexes(deals_df)
        deals_gdf: gpd.GeoDataFrame = Preprocessor.add_geolocation(deals_df)
        deals_gdf: gpd.GeoDataFrame = Preprocessor.remove_missing_rows(deals_gdf)
        self.deals_gdf: gpd.GeoDataFrame = deals_gdf

    @staticmethod
    def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=['Unnamed: 0']).reset_index(drop=True)

    @staticmethod
    def remove_hidden_characters(df: pd.DataFrame) -> pd.DataFrame:
        def remove_hidden_chars(s):
            if isinstance(s, str):
                return s.replace('\u200E', '').replace('\u200F', '')

            return s

        # Apply the function to each string column in the DataFrame
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].apply(remove_hidden_chars)

        return df

    @staticmethod
    def remove_irrelevant_deals(df: pd.DataFrame) -> pd.DataFrame:
        relevant_deal_types: List[str] = [
            'דירת גן', 'דירה בבית קומות', "קוטג' טורי", 'בית בודד', "קוטג' חד משפחתי", 'מגורים', 'חד משפחתי (וילה)',
            'דירה', 'דירת גג (פנטהאוז)', 'מיני פנטהאוז', 'בית פרטי', "קוטג' דו משפחתי", 'דופלקס', 'דירת גג'
        ]
        return df[df[Config.DEAL_TYPE_FIELD_NAME].isin(relevant_deal_types)]

    def add_settlements_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        df[Config.SETTLEMENT_NUMERIC_FIELD_NAME] = df[Config.SETTLEMENT_FIELD_NAME].apply(
            lambda x: self.tokenizer.settlement_str_to_numeric(x))
        return df

    @staticmethod
    def add_numeric_floor_numbers(df: pd.DataFrame) -> pd.DataFrame:
        floor_translations: Dict[str, Dict[str, Any]] = {}
        for filename in os.listdir(Config.FLOORS_TRANSLATIONS_LOCATION):
            with open(Path(Config.FLOORS_TRANSLATIONS_LOCATION, filename), 'r', encoding='utf-8') as f:
                file_data: Dict[str, Dict[str, Any]] = json.load(fp=f)
                floor_translations.update(file_data)

        starting_floor_translations: Dict[str, int] = {
            k: sorted(list(set(v['floors'])))[0] for k, v in floor_translations.items()
            if len(v['floors']) > 0 and v['sequential']
        }
        num_floors_translations: Dict[str, int] = {
            k: len(set(v['floors'])) for k, v in floor_translations.items()
            if len(v['floors']) > 0 and v['sequential']
        }
        df[Config.STARTING_FLOOR_NUMBER_FIELD_NAME] = df[Config.FLOOR_NUMBER_FIELD_NAME].map(starting_floor_translations)
        df[Config.NUMBER_FLOORS_FIELD_NAME] = df[Config.FLOOR_NUMBER_FIELD_NAME].map(num_floors_translations)
        return df

    @staticmethod
    def add_geolocation(df: pd.DataFrame) -> gpd.GeoDataFrame:
        geolocation_data: Dict[str, Dict[str, Any]] = {}
        for filename in os.listdir(Config.GEOCODING_DATA_DIR_LOCATION):
            with open(Path(Config.GEOCODING_DATA_DIR_LOCATION, filename), 'r', encoding='utf-8') as f:
                file_data: Dict[str, Dict[str, Any]] = json.load(fp=f)
                geolocation_data.update(file_data)

        geolocation_data_clean: Dict[str, Tuple[float, float]] = {}
        for key, value in geolocation_data.items():
            if value is None:
                continue

            try:
                x_itm: float = value["data"]["ADDRESS"][0]["X"]
                y_itm: float = value["data"]["ADDRESS"][0]["Y"]
                geolocation_data_clean[key] = (x_itm, y_itm)
            except KeyError:
                continue

        itm_geolocations: pd.Series = df[Config.FULL_ADDRESS_FIELD_NAME].map(geolocation_data_clean).apply(
            lambda x: x if not pd.isna(x) else (np.nan, np.nan)
        )
        df[Config.X_ITM_FIELD_NAME], df[Config.Y_ITM_FIELD_NAME] = zip(*itm_geolocations)
        mask: pd.Series = df[[Config.X_ITM_FIELD_NAME, Config.Y_ITM_FIELD_NAME]].notna().all(axis=1)
        x_itm_clean: pd.Series = df.loc[mask, Config.X_ITM_FIELD_NAME]
        y_itm_clean: pd.Series = df.loc[mask, Config.Y_ITM_FIELD_NAME]
        y_wsg_vector, x_wsg_vector = itm_to_wsg84(x_itm_clean.to_numpy(), y_itm_clean.to_numpy())
        df.loc[mask, Config.X_WSG_FIELD_NAME], df.loc[mask, Config.Y_WSG_FIELD_NAME] = x_wsg_vector, y_wsg_vector
        gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            df, geometry=[
                Point(x, y) if not pd.isna(x) else None
                for x, y in zip(df[Config.X_ITM_FIELD_NAME], df[Config.Y_ITM_FIELD_NAME])
            ]
        )
        gdf.set_crs(crs=Config.ITM_EPSG_CODE, inplace=True)
        return gdf

    @staticmethod
    def remove_hebrew(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf: gpd.GeoDataFrame = gdf.drop(columns=[
            Config.FULL_ADDRESS_FIELD_NAME,
            Config.DISPLAY_ADDRESS_FIELD_NAME,
            Config.TREND_FORMAT_FIELD_NAME,
            Config.SETTLEMENT_FIELD_NAME
        ])
        return gdf

    @staticmethod
    def remove_missing_rows(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        necessary_features_field_names: List[str] = [
            Config.DEAL_DATE_FIELD_NAME,
            Config.DEAL_TYPE_FIELD_NAME,
            Config.ROOMS_NUMBER_FIELD_NAME,
            Config.SIZE_FIELD_NAME,
            Config.DEAL_SUM_FIELD_NAME,
            Config.BUILDING_BUILD_YEAR_FIELD_NAME,
            Config.BUILDING_NUM_FLOORS_FIELD_NAME,
            Config.STARTING_FLOOR_NUMBER_FIELD_NAME,
            Config.NUMBER_FLOORS_FIELD_NAME,
            Config.SETTLEMENT_NUMERIC_FIELD_NAME,
            Config.X_ITM_FIELD_NAME,
            Config.Y_ITM_FIELD_NAME
        ]
        return gdf.dropna(subset=necessary_features_field_names, how='any')

    def init_graph(self) -> None:
        addresses_records: List[Dict[str, Union[str, Point]]] = []
        address_groups = self.deals_gdf.groupby(Config.FULL_ADDRESS_FIELD_NAME)
        for address, address_group in address_groups:
            addresses_records.append({
                "address": address,
                "geometry": Point(
                    (address_group.iloc[0][Config.X_ITM_FIELD_NAME], address_group.iloc[0][Config.Y_ITM_FIELD_NAME]))
            })
        addresses_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame.from_records(data=addresses_records).reset_index(drop=True)
        addresses_gdf.set_geometry("geometry", crs=Config.ITM_EPSG_CODE, inplace=True)

        buffered: gpd.GeoDataFrame = addresses_gdf.copy()
        buffered['geometry'] = buffered.geometry.buffer(self.radius_meters)
        # Find edges using spatial join
        possible_edges = gpd.sjoin(buffered, addresses_gdf, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')  # can't run in debug mode as it runs twice
        possible_edges = possible_edges[possible_edges.index != possible_edges.index_right]  # Filter out self-matches
        # Create a DataFrame from the edges
        edges_df = pd.DataFrame({
            'index_left': possible_edges.index,
            'index_right': possible_edges.index_right
        }).reset_index(drop=True)
        left_geometries: gpd.GeoSeries = addresses_gdf.iloc[edges_df['index_left']].geometry
        right_geometries: gpd.GeoSeries = addresses_gdf.iloc[edges_df['index_right']].geometry
        distances: gpd.GeoSeries = left_geometries.distance(right_geometries, align=False).reset_index(drop=True)
        edges_df: pd.DataFrame = pd.concat([edges_df, distances], axis=1, ignore_index=True)
        edges_df.rename({0: 'index_left', 1: 'index_right', 2: 'distance'}, axis=1, inplace=True)

        self.address_groups = address_groups
        self.addresses_gdf: gpd.GeoDataFrame = addresses_gdf
        self.addresses_edges: pd.DataFrame = edges_df

    def create_final_output(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        addresses_gdfs: List[gpd.GeoDataFrame] = []
        for address_idx, (address, address_gdf) in enumerate(self.address_groups):
            address_gdf[Config.ADDRESS_IDX_FIELD_NAME] = address_idx
            addresses_gdfs.append(address_gdf)

        self.deals_gdf: gpd.GeoDataFrame = pd.concat(addresses_gdfs, axis=0)
        deals_df: pd.DataFrame = pd.DataFrame(
            self.deals_gdf[[
                Config.DEAL_DATE_FIELD_NAME,
                Config.DEAL_TYPE_FIELD_NAME,
                Config.ROOMS_NUMBER_FIELD_NAME,
                Config.SIZE_FIELD_NAME,
                Config.DEAL_SUM_FIELD_NAME,
                Config.BUILDING_BUILD_YEAR_FIELD_NAME,
                Config.BUILDING_NUM_FLOORS_FIELD_NAME,
                Config.STARTING_FLOOR_NUMBER_FIELD_NAME,
                Config.NUMBER_FLOORS_FIELD_NAME,
                Config.SETTLEMENT_NUMERIC_FIELD_NAME,
                Config.ADDRESS_IDX_FIELD_NAME,
                Config.X_ITM_FIELD_NAME,
                Config.Y_ITM_FIELD_NAME,
                Config.X_WSG_FIELD_NAME,
                Config.Y_WSG_FIELD_NAME,
                Config.FULL_ADDRESS_FIELD_NAME,
                Config.DISPLAY_ADDRESS_FIELD_NAME,
                Config.TREND_FORMAT_FIELD_NAME,
                Config.TREND_IS_NEGATIVE_FIELD_NAME,
                Config.SETTLEMENT_FIELD_NAME
            ]]
        )
        deal_types: List[str] = list(set(deals_df[Config.DEAL_TYPE_FIELD_NAME].to_list()))
        deals_df[Config.DEAL_TYPE_NUMERIC_FIELD_NAME] = deals_df[Config.DEAL_TYPE_FIELD_NAME].apply(lambda x: deal_types.index(x))
        deals_df[Config.DEAL_DATE_FIELD_NAME] = pd.to_datetime(deals_df[Config.DEAL_DATE_FIELD_NAME], format='%d.%m.%Y')
        deals_df[Config.DEAL_YEAR_FIELD_NAME] = deals_df[Config.DEAL_DATE_FIELD_NAME].dt.year.astype(int)
        earliest_date: datetime = deals_df[Config.DEAL_DATE_FIELD_NAME].min()

        def months_diff(date):
            return (date.year - earliest_date.year) * 12 + date.month - earliest_date.month

        deals_df[Config.DEAL_DATE_NUMERIC_FIELD_NAME] = deals_df[Config.DEAL_DATE_FIELD_NAME].apply(months_diff).astype(int)
        deals_df[Config.DEAL_SUM_FIELD_NAME] = deals_df[Config.DEAL_SUM_FIELD_NAME].str.replace(',', '').astype(float)
        deals_df.reset_index(drop=True, inplace=True)

        addresses_df: pd.DataFrame = pd.DataFrame(self.addresses_gdf["address"])
        addresses_df["centroid_X_ITM"] = [i.xy[0][0] for i in self.addresses_gdf.geometry]
        addresses_df["centroid_Y_ITM"] = [i.xy[1][0] for i in self.addresses_gdf.geometry]

        return deals_df, addresses_df, self.addresses_edges


if __name__ == '__main__':
    RADIUS: float = 100
    preprocessor: Preprocessor = Preprocessor(radius_meters=RADIUS)
    deals, addresses, addresses_edges = preprocessor.create_final_output()
    deals.to_csv(Config.POST_PROCESSED_DEALS_LOCATION, index=False)
    addresses.to_csv(Config.POST_PROCESSED_ADDRESSES_LOCATION, index=True, index_label='index')
    addresses_edges.to_csv(Config.POST_PROCESSED_EDGES_LOCATION, index=False)
