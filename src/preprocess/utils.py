from typing import Tuple

import numpy as np
from pyproj import Transformer

from src.config import Config


transformer = Transformer.from_crs(f'epsg:{Config.ITM_EPSG_CODE}', f'epsg:{Config.WSG84_EPSG_CODE}', always_xy=True)


def itm_to_wsg84(x_itm_vector: np.ndarray, y_itm_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: (y_wsg_vector, x_wsg_vector)
    """
    return transformer.transform(x_itm_vector, y_itm_vector)


# itm = Proj(init=Config.ITM_EPSG_CODE)  # Israel Transverse Mercator system
# wgs84 = Proj(init=Config.WSG84_EPSG_CODE)  # WGS 84 system


# def itm_to_wsg84(x_itm_vector: np.ndarray, y_itm_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     x_wsg_vector, y_wsg_vector = [], []
#     for x_itm, y_itm in tqdm(zip(x_itm_vector, y_itm_vector)):
#         x_wsg, y_wsg = transform(itm, wgs84, x_itm, y_itm)
#         x_wsg_vector.append(x_wsg)
#         y_wsg_vector.append(y_wsg)
#
#     return np.ndarray(x_wsg_vector), np.ndarray(y_wsg_vector)
