from sodapy import Socrata
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from os import getenv


def get_api_df():
    cols = ['OSEBuildingID', 'DataYear', 'BuildingType', 'PrimaryPropertyType',
       'PropertyName', 'Address', 'City', 'State', 'ZipCode',
       'TaxParcelIdentificationNumber', 'CouncilDistrictCode', 'Neighborhood',
       'Latitude', 'Longitude', 'YearBuilt', 'NumberofBuildings',
       'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking',
       'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes',
       'LargestPropertyUseType', 'LargestPropertyUseTypeGFA',
       'SecondLargestPropertyUseType', 'SecondLargestPropertyUseTypeGFA',
       'ThirdLargestPropertyUseType', 'ThirdLargestPropertyUseTypeGFA',
       'YearsENERGYSTARCertified', 'ENERGYSTARScore', 'SiteEUI(kBtu/sf)',
       'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)',
       'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)', 'SteamUse(kBtu)',
       'Electricity(kWh)', 'Electricity(kBtu)', 'NaturalGas(therms)',
       'NaturalGas(kBtu)', 'DefaultData', 'Comments', 'ComplianceStatus',
       'Outlier', 'TotalGHGEmissions', 'GHGEmissionsIntensity']
    
    
    client = Socrata("data.seattle.gov", None)
    results = client.get("2bpz-gwpy", 'csv', limit=10000)
    df = pd.DataFrame.from_records(data=results[1:], columns=cols)
    return df

def format_df(df):
    df = df.replace('', np.nan)
    types = {
        'OSEBuildingID':                      object,
        'DataYear':                           int,
        'BuildingType':                       object,
        'PrimaryPropertyType':                object,
        'PropertyName':                       object,
        'Address':                            object,
        'City':                               object,
        'State':                              object,
        'ZipCode':                            float,
        'TaxParcelIdentificationNumber':      object,
        'CouncilDistrictCode':                int,
        'Neighborhood':                       object,
        'Latitude':                           float,
        'Longitude':                          float,
        'YearBuilt':                          int,
        'NumberofBuildings':                  float,
        'NumberofFloors':                     int,
        'PropertyGFATotal':                   int,
        'PropertyGFAParking':                 int,
        'PropertyGFABuilding(s)':             int,
        'ListOfAllPropertyUseTypes':          object,
        'LargestPropertyUseType':             object,
        'LargestPropertyUseTypeGFA':          float,
        'SecondLargestPropertyUseType':       object,
        'SecondLargestPropertyUseTypeGFA':    float,
        'ThirdLargestPropertyUseType':        object,
        'ThirdLargestPropertyUseTypeGFA':     float,
        'YearsENERGYSTARCertified':           object,
        'ENERGYSTARScore':                    float,
        'SiteEUI(kBtu/sf)':                   float,
        'SiteEUIWN(kBtu/sf)':                 float,
        'SourceEUI(kBtu/sf)':                 float,
        'SourceEUIWN(kBtu/sf)':               float,
        'SiteEnergyUse(kBtu)':                float,
        'SiteEnergyUseWN(kBtu)':              float,
        'SteamUse(kBtu)':                     float,
        'Electricity(kWh)':                   float,
        'Electricity(kBtu)':                  float,
        'NaturalGas(therms)':                 float,
        'NaturalGas(kBtu)':                   float,
        'DefaultData':                        bool,
        'Comments':                           float,
        'ComplianceStatus':                   object,
        'Outlier':                            object,
        'TotalGHGEmissions':                  float,
        'GHGEmissionsIntensity':              float,
    }
    
    df = df.replace('', np.nan)
    df = df.replace('NULL', np.nan)
    df = df.astype(types)
    return df


def get_engine(echo_arg):

    url = getenv('POSTGRESQLCONNSTR_URL')

    engine = create_engine(url, echo=echo_arg)
    return engine


