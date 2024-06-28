from typing import List, Dict, Any, Tuple

import matplotlib
# matplotlib.use('TkAgg')
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap
from shapely import Point
from tqdm import tqdm
import folium
from branca.colormap import linear

from src.config import Config
from src.preprocess.utils import itm_to_wsg84


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
MAXIMUM_EDGE_WEIGHT: float = 2
MAXIMUM_PATH_WEIGHT: float = 10
MAXIMUM_DEAL_AMOUNT: float = 40000000


def add_edges_graph(graph: nx.DiGraph, dst: gpd.GeoDataFrame, src: gpd.GeoDataFrame) -> None:
    src: gpd.GeoDataFrame = src.copy()
    src.geometry = src.geometry.buffer(RADIUS_METERS)
    joined: gpd.GeoDataFrame = gpd.sjoin(dst, src, how='inner', predicate='intersects')
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
    edges_df: pd.DataFrame = pd.DataFrame({
        'target_idx': joined.index,
        'source_idx': joined.index_right
    })
    edges_df['weight'] = (-1) * np.log(inflation)
    min_threshold: float = (-1) * np.log(MAXIMUM_EDGE_WEIGHT)
    print(f"number of edges: {len(edges_df)}")
    for _, edge in tqdm(edges_df.iterrows(), total=len(edges_df)):
        if edge['weight'] < min_threshold:
            continue

        graph.add_edge(edge['source_idx'], edge['target_idx'], weight=edge['weight'])


def create_layered_graph(data: gpd.GeoDataFrame) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()
    for i, deal in tqdm(data.iterrows(), total=len(data)):
        graph.add_node(i, **deal)

    for year in range(1999, 2023):
        print(f"year {year}")
        current_year_deals: gpd.GeoDataFrame = data[data[Config.DEAL_YEAR_FIELD_NAME] == year]
        last_year_deals: gpd.GeoDataFrame = data[data[Config.DEAL_YEAR_FIELD_NAME] == (year-1)]
        add_edges_graph(graph, current_year_deals, last_year_deals)

    return graph


def find_all_pairs_shortest_paths(graph: nx.DiGraph) -> List[Tuple[List[int], float]]:
    all_shortest_paths: Dict[int, Dict[int, Any]] = dict(nx.all_pairs_bellman_ford_path(graph, weight='weight'))
    paths: List[Tuple[List[int], float]] = []
    for i, v in all_shortest_paths.items():
        for j, path in v.items():
            if i == j:
                continue

            path: List[int] = [int(k) for k in path]
            path_weight: float = np.exp((-1) * sum([graph[u][t]['weight'] for u, t in zip(path[:-1], path[1:])]))
            paths.append((path, path_weight))

    return sorted(paths, key=lambda x: x[1], reverse=True)


def find_centrality(graph: nx.DiGraph, shortest_paths: List[Tuple[List[int], float]]) -> Dict[int, float]:
    centrality: Dict[int, float] = {i: 0 for i in graph.nodes}
    for path, path_weight in shortest_paths:
        for u in path:
            # centrality[u] += 1
            centrality[u] += path_weight

    return centrality


def find_hot_regions() -> None:
    deals_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_DEALS_LOCATION)
    deals_df: pd.DataFrame = deals_df[deals_df[Config.DEAL_SUM_FIELD_NAME] <= 40000000]
    deals_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        deals_df,
        geometry=[
            Point(x, y) if (not pd.isna(x)) and (not pd.isna(y)) else None
            for x, y in zip(deals_df[Config.X_ITM_FIELD_NAME], deals_df[Config.Y_ITM_FIELD_NAME])
        ]
    )
    deals_gdf.set_crs(Config.ITM_EPSG_CODE, inplace=True)
    graph: nx.DiGraph = create_layered_graph(deals_gdf)
    shortest_paths: List[Tuple[List[int], float]] = find_all_pairs_shortest_paths(graph)
    idx: int = 0
    for idx, (path, path_weight) in enumerate(shortest_paths):
        if path_weight <= MAXIMUM_PATH_WEIGHT:
            break

    shortest_paths: List[Tuple[List[int], float]] = shortest_paths[idx:]
    centrality: Dict[int, float] = find_centrality(graph, shortest_paths)
    nx.set_node_attributes(graph, centrality, "centrality")
    importance_df = pd.DataFrame.from_records([{
        'X_ITM': graph.nodes[i]['X_ITM'],
        'Y_ITM': graph.nodes[i]['Y_ITM'],
        'centrality': graph.nodes[i]['centrality']
    } for i in graph.nodes])
    # importance_df.sort_values(by='centrality', ascending=False, inplace=True)
    # Convert X_ITM, Y_ITM to latitude and longitude if needed (assuming it's already lat/lon here)
    importance_df['X_WSG84'], importance_df['Y_WSG84'] = itm_to_wsg84(importance_df['X_ITM'].values,
                                                                      importance_df['Y_ITM'].values)
    # Normalize 'centrality' column
    importance_df['centrality_normalized'] = (importance_df['centrality'] - importance_df['centrality'].min()) / (
                importance_df['centrality'].max() - importance_df['centrality'].min())
    importance_df.to_csv('importance.csv')

    # Setup the initial location and the map
    map_israel = folium.Map(location=[32.0853, 34.7818], zoom_start=10)  # Centered around Tel Aviv for start

    # Add a heatmap to the map
    heatmap_data = importance_df[['Y_WSG84', 'X_WSG84', 'centrality_normalized']].values.tolist()  # Ensure coordinates are lat, lon order if needed
    HeatMap(heatmap_data, radius=15).add_to(map_israel)

    # Save to HTML
    map_israel.save('Israel_Heatmap_centrality_1.html')


if __name__ == '__main__':
    find_hot_regions()
