from typing import Dict, Any, Tuple, Union, Optional
import logging
import json
import os

import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely import Point

from src.config import Config
from src.preprocess.utils import itm_to_wsg84


logging.basicConfig(level=logging.INFO)


class Preprocessor:
    @staticmethod
    def load_data(radius: Optional[float]) -> Union[Tuple[gpd.GeoDataFrame, pd.DataFrame], gpd.GeoDataFrame]:
        df: pd.DataFrame = pd.read_csv(Config.DATA_LOCATION_FILEPATH)
        df: pd.DataFrame = df.replace('nan', np.NAN)
        df: pd.DataFrame = df.reset_index(drop=False)
        df: pd.DataFrame = Preprocessor.remove_unnecessary_columns(df)
        # df: pd.DataFrame = Preprocessor.remove_hebrew(df)
        df: pd.DataFrame = Preprocessor.add_geolocation(df)
        df: pd.DataFrame = Preprocessor.remove_hidden_characters(df)
        gdf: gpd.GeoDataFrame = Preprocessor.add_geometry(df)
        if radius:
            pairs: pd.DataFrame = Preprocessor.create_edges(gdf, radius)
            return gdf, pairs

        # else (no radius was given, meaning we don't need to calculate any edges)
        return gdf

    @staticmethod
    def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=['Unnamed: 0', 'index'])

    @staticmethod
    def add_geolocation(df: pd.DataFrame) -> pd.DataFrame:
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
            lambda x: x if not pd.isna(x) else (np.NAN, np.NAN)
        )
        df[Config.X_ITM_FIELD_NAME], df[Config.Y_ITM_FIELD_NAME] = zip(*itm_geolocations)
        mask: pd.Series = df[[Config.X_ITM_FIELD_NAME, Config.Y_ITM_FIELD_NAME]].notna().all(axis=1)
        x_itm_clean: pd.Series = df.loc[mask, Config.X_ITM_FIELD_NAME]
        y_itm_clean: pd.Series = df.loc[mask, Config.Y_ITM_FIELD_NAME]
        y_wsg_vector, x_wsg_vector = itm_to_wsg84(x_itm_clean.to_numpy(), y_itm_clean.to_numpy())
        df.loc[mask, Config.X_WSG_FIELD_NAME], df.loc[mask, Config.Y_WSG_FIELD_NAME] = x_wsg_vector, y_wsg_vector

        return df

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
    def remove_hebrew(df: pd.DataFrame) -> pd.DataFrame:
        return df

    @staticmethod
    def add_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
        gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            df, geometry=[
                Point(x, y) if not pd.isna(x) else None
                for x, y in zip(df[Config.X_ITM_FIELD_NAME], df[Config.Y_ITM_FIELD_NAME])
            ]
        )
        gdf.set_crs(crs=Config.ITM_EPSG_CODE, inplace=True)

        return gdf

    @staticmethod
    def create_edges(gdf: gpd.GeoDataFrame, radius_meters: float) -> pd.DataFrame:
        buffered: gpd.GeoDataFrame = gdf.copy()
        buffered['geometry'] = buffered.geometry.buffer(radius_meters)
        # Find pairs using spatial join
        possible_pairs = gpd.sjoin(buffered, gdf, how='inner', op='intersects', lsuffix='left', rsuffix='right')  # can't run in debug mode as it runs twice
        possible_pairs = possible_pairs[possible_pairs.index != possible_pairs.index_right]  # Filter out self-matches
        # Create a DataFrame from the pairs
        pairs_df = pd.DataFrame({
            'index_left': possible_pairs.index,
            'index_right': possible_pairs.index_right
        }).reset_index(drop=True)
        left_geometries: gpd.GeoSeries = gdf.iloc[pairs_df['index_left']].geometry
        right_geometries: gpd.GeoSeries = gdf.iloc[pairs_df['index_right']].geometry
        distances: gpd.GeoSeries = left_geometries.distance(right_geometries, align=False).reset_index(drop=True)
        pairs_df: pd.DataFrame = pd.concat([pairs_df, distances], axis=1, ignore_index=True)
        pairs_df.rename({0: 'index_left', 1: 'index_right', 2: 'distance'}, axis=1, inplace=True)

        return pairs_df


if __name__ == '__main__':
    RADIUS: float = 20
    vertices, edges = Preprocessor.load_data(radius=RADIUS)
    vertices.to_csv(Config.VERTICES_FILEPATH, index_label=True)
    edges.to_csv(f'{Config.DATA_DIRPATH}/edges_radius_{RADIUS}.csv', index=False)
