from typing import Tuple, Any, Dict

import requests
from requests import Response

from src.config import GoogleMapsAPIConfig


def get_geolocation(address: str) -> Tuple[float, float]:
    """
    :param address: The address to search the geolocation for.
    :return: (latitude, longitude).
    """
    url = GoogleMapsAPIConfig.GOOGLE_MAPS_API_URL
    response: Response = requests.get(
        url, params={
            GoogleMapsAPIConfig.GOOGLE_MAPS_API_KEY_FIELD_NAME: GoogleMapsAPIConfig.GOOGLE_MAPS_API_KEY,
            GoogleMapsAPIConfig.GOOGLE_MAPS_API_ADDRESS_FIELD_NAME: address
        }, headers={}, data={}
    )
    result: Dict[str, Any] = response.json()
    location: Dict[str, float] = result.get("results")[0].get("geometry").get("location")
    return location.get("lat"), location.get("lng")

