import os
from typing import List, Tuple, Set
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm

from src.preprocess.preprocessor import Preprocessor
from src.config import Config


logging.basicConfig(level=logging.INFO)


class Analyzer:
    def __init__(self, edges_radius: float):
        self.edges_radius: float = edges_radius
        logging.info(f"Radius={edges_radius}")
        edges_filepath: str = f'{Config.DATA_DIRPATH}/edges_radius_{edges_radius}.csv'
        if os.path.isfile(edges_filepath):
            logging.info("Edges file was found in filesystem, loading it")
            self.edges: pd.DataFrame = pd.read_csv(edges_filepath)
            self.vertices: gpd.GeoDataFrame = Preprocessor.load_data(radius=None, settlements_subset=Config.SETTLEMENTS_SUBSET)
        else:
            logging.info("Edges file wasn't found in filesystem, deriving the edges")
            self.vertices, self.edges = Preprocessor.load_data(radius=edges_radius, settlements_subset=Config.SETTLEMENTS_SUBSET)
            self.edges.to_csv(edges_filepath, index=False)

        logging.info(f"# Edges: {len(self.edges)}")
        logging.info("Creating the graph")
        self.graph: nx.Graph = nx.Graph()
        self.create_graph()
        logging.info("Calculating connected components")
        connected_components: List[Set[int]] = list(nx.connected_components(self.graph))
        connected_components_lengths: List[int] = [len(ce) for ce in connected_components]
        connected_component_lengths_count: List[Tuple[int, int]] = \
            [(i, connected_components_lengths.count(i)) for i in set(connected_components_lengths)]
        logging.info(f"Connected components lengths count: {connected_component_lengths_count}")
        pass

    def create_graph(self):
        vertices_indices: List[int] = self.vertices[~pd.isna(self.vertices.geometry)].index.to_list()
        self.graph.add_nodes_from(vertices_indices)
        edges: List[Tuple[int, int, float]] = zip(
            self.edges['index_left'],
            self.edges['index_right'],
            (self.edges_radius - self.edges['distance']) / self.edges_radius
        )
        self.graph.add_weighted_edges_from(edges)


if __name__ == '__main__':
    analyzer = Analyzer(edges_radius=50)
