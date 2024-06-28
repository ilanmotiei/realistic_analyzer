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

    POST_PROCESSED_ADDRESSES_LOCATION: str = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/" \
                                        "Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data/24062024/" \
                                            "addresses.csv"
    POST_PROCESSED_DEALS_LOCATION: str = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/" \
                                        "Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data/24062024/" \
                                            "deals.csv"
    POST_PROCESSED_EDGES_LOCATION: str = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/" \
                                         "Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data/24062024/" \
                                         "addresses_edges.csv"

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
    STARTING_FLOOR_NUMBER_FIELD_NAME: str = "start_floor"
    NUMBER_FLOORS_FIELD_NAME: str = "num_floors"
    SETTLEMENT_NUMERIC_FIELD_NAME: str = "settlement_numeric"
    ADDRESS_IDX_FIELD_NAME: str = "address_idx"
    DEAL_YEAR_FIELD_NAME: str = "year"
    DEAL_TYPE_NUMERIC_FIELD_NAME: str = "deal_type_numeric"
    DEAL_DATE_NUMERIC_FIELD_NAME: str = "deal_date_numeric"

    # other
    ITM_EPSG_CODE: str = "2039"
    WSG84_EPSG_CODE: str = "4326"

    DATA_DIRPATH = "/Users/ilanmotiei/Desktop/University/לימודים/מנהל עסקים/Networked Markets & Graph Analysis/real_estate_analysis/analyzer/data"
    VERTICES_FILEPATH = f'{DATA_DIRPATH}/vertices.csv'
    EDGES_FILEPATH = f'{DATA_DIRPATH}/edges.csv'

    SETTLEMENTS_SUBSET: List[str] = ['תל אביב']
