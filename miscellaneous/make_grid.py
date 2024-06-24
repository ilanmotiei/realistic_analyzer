import geopandas as gpd
import numpy as np
from shapely.geometry import box

def create_grid(min_lon, max_lon, min_lat, max_lat, km_in_deg):
    grid_cells = []
    for lon in np.arange(min_lon, max_lon, km_in_deg):
        for lat in np.arange(min_lat, max_lat, km_in_deg):
            # Create a 1km x 1km square polygon
            grid_cells.append(box(lon, lat, lon + km_in_deg, lat + km_in_deg))
    return grid_cells

# Define bounding box coordinates for Tel Aviv
min_lon, max_lon = 34.7, 34.9
min_lat, max_lat = 32.0, 32.2
km_in_deg = 1 / 111  # Approximation of 1 km in degree at Tel Aviv latitude

# Create grid
grid_cells = create_grid(min_lon, max_lon, min_lat, max_lat, km_in_deg)

# Create GeoDataFrame
grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")

# Export to a shapefile
grid_gdf.to_file("TelAviv_1km_Grid.shp")
