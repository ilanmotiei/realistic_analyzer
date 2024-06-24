from typing import List


class Config:
    DATA_LOCATION_FILEPATH: str = "/Users/ilanmotiei/Desktop/gov_scraper/results/parsed_results.csv"
    GEOCODING_DATA_DIR_LOCATION: str = "/Users/ilanmotiei/Desktop/gov_scraper/geolocation_results/pages/"
    FLOORS_TRANSLATIONS_LOCATION: str = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/" \
                                        "Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data/" \
                                        "floors_translations_pages_gpt4_batch/"
    ALL_ISRAELI_SETTLEMENTS_LOCATION: str = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/" \
                                        "Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data/" \
                                        "all_israeli_settlements.json"

    # Original fields names
    DEAL_DATE_FIELD_NAME: str = "DEALDATE"
    DEAL_DATETIME_FIELD_NAME: str = "DEALDATETIME"
    FULL_ADDRESS_FIELD_NAME: str = "FULLADRESS"
    DISPLAY_ADDRESS_FIELD_NAME: str = "DISPLAYADRESS"
    GUSH_FIELD_NAME: str = "GUSH"
    DEAL_TYPE_FIELD_NAME: str = "DEALNATUREDESCRIPTION"
    ROOMS_NUMBER_FIELD_NAME: str = "ASSETROOMNUM"
    FLOOR_NUMBER_FIELD_NAME: str = "FLOORNO"
    SIZE_FIELD_NAME: str = "DEALNATURE"
    DEAL_SUM_FIELD_NAME: str = "DEALAMOUNT"
    NEW_PROJECT_FIELD_NAME: str = "NEWPROJECTTEXT"
    PROJECT_FIELD_NAME: str = "PROJECTNAME"
    BUILDING_BUILD_YEAR_FIELD_NAME: str = "BUILDINGYEAR"  # The year the building the asset is included at was founded
    BUILD_YEAR_FIELD_NAME: str = "YEARBUILT"  # When was the asset built at (may differ from the year of the building foundation)
    BUILDING_NUM_FLOORS_FIELD_NAME: str = "BUILDINGFLOORS"  # Amount of floors in the building
    KEY_VALUE_FIELD_NAME: str = "KEYVALUE"  # Not relevant
    ASSET_TYPE_FIELD_NAME: str = "TYPE"
    POLYGON_ID_FIELD_NAME: str = "POLYGON_ID"
    TREND_IS_NEGATIVE_FIELD_NAME: str = "TREND_IS_NEGATIVE"
    TREND_FORMAT_FIELD_NAME: str = "TREND_FORMAT"
    SETTLEMENT_FIELD_NAME: str = "settlement"

    # Calculated fields names
    X_ITM_FIELD_NAME: str = "X_ITM"
    Y_ITM_FIELD_NAME: str = "Y_ITM"
    X_WSG_FIELD_NAME: str = "X_WSG84"
    Y_WSG_FIELD_NAME: str = "Y_WSG84"
    STARTING_FLOOR_NUMBER_FIELD_NAME: str = "STARTING_FLOOR"
    NUMBER_FLOORS_FIELD_NAME: str = "NUM_FLOORS"
    SETTLEMENT_NUMERIC_FIELD_NAME: str = "SETTLEMENT_NUMERIC"
    ADDRESS_IDX_FIELD_NAME: str = "address_idx"

    # other
    ITM_EPSG_CODE: str = "2039"
    WSG84_EPSG_CODE: str = "4326"

    DATA_DIRPATH = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data"
    VERTICES_FILEPATH = f'{DATA_DIRPATH}/vertices.csv'
    EDGES_FILEPATH = f'{DATA_DIRPATH}/edges.csv'

    SETTLEMENTS_SUBSET: List[str] = ['תל אביב']
