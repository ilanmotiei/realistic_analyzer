from typing import Dict, List, Hashable, Optional

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import Point
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import Config


grid_wsg: gpd.GeoDataFrame = gpd.read_file('../../data/Israel_1km_Grid/Israel_1km_Grid.shp')
grid_wsg.set_crs(Config.WSG84_EPSG_CODE, inplace=True)
grid_itm: gpd.GeoDataFrame = grid_wsg.to_crs(Config.ITM_EPSG_CODE)
deals_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_DEALS_LOCATION)
deals: gpd.GeoDataFrame = gpd.GeoDataFrame(
    deals_df,
    geometry=[
        Point(x, y) if (not pd.isna(x)) and (not pd.isna(y)) else None
        for x, y in zip(deals_df[Config.X_ITM_FIELD_NAME], deals_df[Config.Y_ITM_FIELD_NAME])
    ]
)
deals: gpd.GeoDataFrame = deals[deals[Config.DEAL_YEAR_FIELD_NAME] <= 2023]
deals.set_crs(Config.ITM_EPSG_CODE, inplace=True)

deals_gridded: gpd.GeoDataFrame = gpd.sjoin(deals, grid_itm, how='inner', op='intersects')

stats: Dict[Hashable, Dict[Hashable, Dict[Hashable, Optional[float]]]] = {
    year: {
        segment_id: {
            room_amount: None for room_amount in set(deals_gridded[Config.ROOMS_NUMBER_FIELD_NAME])
        } for segment_id in tqdm(set(grid_itm['FID']))
    } for year in tqdm(set(deals_gridded[Config.DEAL_YEAR_FIELD_NAME]))
}

for year, year_df in tqdm(deals_gridded.groupby(deals_gridded[Config.DEAL_DATE_FIELD_NAME].dt.year)):
    for segment_id, segment_year_df in year_df.groupby('FID'):
        for rooms_amount, rooms_amount_df in segment_year_df.groupby(Config.ROOMS_NUMBER_FIELD_NAME):
            avg_price_per_meter_df: pd.DataFrame = rooms_amount_df[Config.DEAL_SUM_FIELD_NAME] / rooms_amount_df[Config.SIZE_FIELD_NAME]
            avg_price_per_meter: float = float(np.nanmean(avg_price_per_meter_df.to_numpy()))
            stats[year][segment_id][rooms_amount] = avg_price_per_meter

inflation_rates: List[float] = []
for year, year_stats in tqdm(stats.items()):
    try:
        prev_year_stats: Dict[Hashable, Dict[Hashable, float]] = stats[year-1]
    except KeyError:
        continue

    inflation_sum: float = 0
    inflation_count: int = 0
    for segment_id, segment_data in year_stats.items():
        prev_year_segment_data: pd.DataFrame = prev_year_stats[segment_id]

        for rooms_amount, rooms_avg_price_per_meter in segment_data.items():
            if rooms_avg_price_per_meter is None:
                continue

            prev_year_segment_rooms_avg_price_per_meter: Optional[float] = prev_year_segment_data[rooms_amount]
            if prev_year_segment_rooms_avg_price_per_meter is None:
                continue

            inflation_sum += rooms_avg_price_per_meter / prev_year_segment_rooms_avg_price_per_meter
            inflation_count += 1

    inflation: float = inflation_sum / inflation_count
    inflation_rates.append(inflation)
    print(f"Year {year} Inflation was: {inflation}")


# --- cumulative inflation plot ---

# Years from 1998 to 2023
years = list(range(1998, 2024))

# Calculate cumulative inflation relative to 2005 (base year)
cumulative_inflation = []
base = 100  # Starting point (2005 as 100 points)
for i in range(len(inflation_rates)):
    if i == 0:
        cumulative_inflation.append(base * inflation_rates[i])
    else:
        cumulative_inflation.append(cumulative_inflation[i - 1] * inflation_rates[i])

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(years, cumulative_inflation, marker='o')
plt.title('Cumulative Inflation Indexed to 1998')
plt.xlabel('Year')
plt.ylabel('Index Points (2005 = 100)')
plt.grid(True)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

# --- inflation plot ---

# Plotting annual inflation rates
plt.figure(figsize=(10, 5))
plt.plot(years, inflation_rates, marker='o', color='red')
plt.title('Annual Inflation Rates (2006-2023)')
plt.xlabel('Year')
plt.ylabel('Inflation Factor (Relative to Previous Year)')
plt.grid(True)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()
