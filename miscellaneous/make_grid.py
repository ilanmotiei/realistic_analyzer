import geopandas as gpd
import numpy as np
from shapely.geometry import box


def create_grid(min_lon, max_lon, min_lat, max_lat, km_in_deg):
    grid_cells = []
    for lon in np.arange(min_lon, max_lon, km_in_deg):
        for lat in np.arange(min_lat, max_lat, km_in_deg):
            # Create a square polygon for each grid cell
            grid_cells.append(box(lon, lat, lon + km_in_deg, lat + km_in_deg))
    return grid_cells


# Define bounding box coordinates for the whole of Israel
min_lon, max_lon = 34.3, 35.9
min_lat, max_lat = 29.5, 33.3

# Approximate conversion of 1 km to degrees of latitude (varies slightly with longitude)
km_in_deg_lat = 1 / 111
# More precise approximation for longitude, adjusting for the cosine of the average latitude
average_lat = (min_lat + max_lat) / 2
km_in_deg_lon = km_in_deg_lat / np.cos(np.radians(average_lat))

# Create grid
grid_cells = create_grid(min_lon, max_lon, min_lat, max_lat, km_in_deg_lon)

# Create GeoDataFrame
grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")

# Export to a shapefile
grid_gdf.to_file("Israel_1km_Grid.shp")
