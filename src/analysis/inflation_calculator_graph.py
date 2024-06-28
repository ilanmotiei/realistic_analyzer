from typing import List, Dict, Any, Tuple

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
import matplotlib.pyplot as plt

from src.config import Config


RADIUS_METERS: float = 100
CRITERIA: List[Dict[str, Any]] = [
    {
        "field_name": Config.ROOMS_NUMBER_FIELD_NAME,
        "max_diff": 0
    },
    {
        "field_name": Config.NUMBER_FLOORS_FIELD_NAME,
        "max_diff": 0
    },
    {
        "field_name": Config.BUILDING_BUILD_YEAR_FIELD_NAME,
        "max_diff": 10
    },
    {
        "field_name": Config.DEAL_TYPE_FIELD_NAME,
        "max_diff": 0
    },
    {
        "field_name": Config.BUILDING_NUM_FLOORS_FIELD_NAME,
        "max_diff": 3
    },
    {
        "field_name": Config.STARTING_FLOOR_NUMBER_FIELD_NAME,
        "max_diff": 0
    }
]

MAXIMUM_YEARLY_INFLATION: float = 2


def calc_yearly_inflation(current: gpd.GeoDataFrame, previous: gpd.GeoDataFrame) -> Tuple[float, int]:
    previous: gpd.GeoDataFrame = previous.copy()
    previous.geometry = previous.geometry.buffer(RADIUS_METERS)
    joined: gpd.GeoDataFrame = gpd.sjoin(current, previous, how='inner', predicate='intersects')
    joined: gpd.GeoDataFrame = joined[joined.index != joined.index_right]  # Filter out self-matches
    # keep only similar assets connected
    for criteria in CRITERIA:
        left_field_name: str = f"{criteria['field_name']}_left"
        right_field_name: str = f"{criteria['field_name']}_right"
        if criteria['max_diff'] <= 0:
            joined: gpd.GeoDataFrame = joined[joined[left_field_name] == joined[right_field_name]]
        else:
            diff: pd.Series = joined[left_field_name] - joined[right_field_name]
            joined: gpd.GeoDataFrame = joined[
                ((-1) * criteria['max_diff'] <= diff) & (diff <= criteria['max_diff'])
            ]

    current_price_per_meter: pd.Series = (joined[f'{Config.DEAL_SUM_FIELD_NAME}_left'] / joined[f'{Config.SIZE_FIELD_NAME}_left'])
    previous_price_per_meter: pd.Series = (joined[f'{Config.DEAL_SUM_FIELD_NAME}_right'] / joined[f'{Config.SIZE_FIELD_NAME}_right'])
    inflation: pd.Series = current_price_per_meter / previous_price_per_meter
    avg_inflation: np.ndarray = np.mean(inflation[inflation <= MAXIMUM_YEARLY_INFLATION])
    return float(avg_inflation), len(joined)


if __name__ == '__main__':
    deals_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_DEALS_LOCATION)
    deals_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        deals_df,
        geometry=[
            Point(x, y) if (not pd.isna(x)) and (not pd.isna(y)) else None
            for x, y in zip(deals_df[Config.X_ITM_FIELD_NAME], deals_df[Config.Y_ITM_FIELD_NAME])
        ]
    )
    deals_gdf.set_crs(Config.ITM_EPSG_CODE, inplace=True)
    inflation_rates: List[float] = []
    num_observations: List[int] = []
    for year in range(1999, 2023):
        current_year_deals: gpd.GeoDataFrame = deals_gdf[deals_df[Config.DEAL_YEAR_FIELD_NAME] == year]
        last_year_deals: gpd.GeoDataFrame = deals_gdf[deals_df[Config.DEAL_YEAR_FIELD_NAME] == (year-1)]
        yearly_inflation, yearly_num_obs = calc_yearly_inflation(current_year_deals, last_year_deals)
        inflation_rates.append(yearly_inflation)
        num_observations.append(yearly_num_obs)

    # Years from 1998 to 2023
    years = list(range(1998, 2022))

    # Calculate cumulative inflation relative to 2005 (base year)
    cumulative_inflation = []
    base = 100  # Starting point (2005 as 100 points)
    for i in range(len(inflation_rates)):
        if i == 0:
            cumulative_inflation.append(base * inflation_rates[i])
        else:
            cumulative_inflation.append(cumulative_inflation[i-1] * inflation_rates[i])

    # --- inflation plot ---

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(years, cumulative_inflation, marker='o')
    plt.title('Cumulative Inflation Indexed to 1998')
    plt.xlabel('Year')
    plt.ylabel('Index Points (1998 = 100)')
    plt.grid(True)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()

    # Plotting inflation rates with number of observations as labels
    plt.figure(figsize=(10, 5))
    plt.plot(years, [(i-1)*100 for i in inflation_rates], 'g-o', label='Annual Inflation Rate (%)')

    # Adding number of observations on top of each data point
    for i, txt in enumerate(num_observations):
        plt.annotate(txt, (years[i], (inflation_rates[i]-1)*100), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Annual Inflation Rate (1999-2022)')
    plt.xlabel('Year')
    plt.ylabel('Inflation Rate (%)')
    plt.xticks(years, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


