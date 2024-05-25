class Config:
    DATA_LOCATION_FILEPATH: str = "/Users/ilanmotiei/Desktop/gov_scraper/results/parsed_results.csv"

    # Original fields names
    DEAL_DATE_FIELD_NAME: str = "DEALDATE"
    DEAL_DATETIME_FIELD_NAME: str = "DEALDATETIME"
    FULL_ADDRESS_FIELD_NAME: str = "FULLADRESS"
    DISPLAY_ADDRESS_FIELD_NAME: str = "DISPLAYADRESS"
    GUSH_FIELD_NAME: str = "GUSH"
    DEAL_DESCRIPTION_FIELD_NAME: str = "DEALNATUREDESCRIPTION"
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
    LATITUDE_FIELD_NAME: str = "latitude"
    LONGITUDE_FIELD_NAME: str = "longitude"

