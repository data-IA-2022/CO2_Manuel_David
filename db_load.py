import pandas as pd
from utils import get_api_df, format_df, get_engine
from sqlalchemy import ( 
    Integer,
    String,  
    Float, 
    Boolean, 
    SmallInteger,
    Text,
    text
)

def main():
    
    engine = get_engine(echo_arg=True)

    df = format_df(get_api_df())
        
    types = {
        'OSEBuildingID': String(length=50226),
        'DataYear': SmallInteger,
        'BuildingType': String(length=20),
        'PrimaryPropertyType': String(length=27),
        'PropertyName': String(length=72),
        'Address': String(length=41),
        'City': String(length=7),
        'State': String(length=2),
        'ZipCode': Float,
        'TaxParcelIdentificationNumber': String(length=25),
        'CouncilDistrictCode': SmallInteger,
        'Neighborhood': String(length=22),
        'Latitude': Float,
        'Longitude': Float,
        'YearBuilt': SmallInteger,
        'NumberofBuildings': SmallInteger,
        'NumberofFloors': SmallInteger,
        'PropertyGFATotal': Integer,
        'PropertyGFAParking': Integer,
        'PropertyGFABuildings': Integer,
        'ListOfAllPropertyUseTypes': String(length=255),
        'LargestPropertyUseType': String(length=52),
        'LargestPropertyUseTypeGFA': Float,
        'SecondLargestPropertyUseType': String(length=52),
        'SecondLargestPropertyUseTypeGFA': Float,
        'ThirdLargestPropertyUseType': String(length=52),
        'ThirdLargestPropertyUseTypeGFA': Float,
        'YearsENERGYSTARCertified': String(length=60),
        'ENERGYSTARScore': SmallInteger,
        'SiteEUI': Float,
        'SiteEUIWN': Float,
        'SourceEUI': Float,
        'SourceEUIWN': Float,
        'SiteEnergyUse': Float,
        'SiteEnergyUseWN': Float,
        'SteamUse': Float,
        'Electricity_kWh': Float,
        'Electricity_kBtu': Float,
        'NaturalGas_therms': Float,
        'NaturalGas_kBtu': Float,
        'DefaultData': Boolean,
        'Comments': Text,
        'ComplianceStatus': String(length=28),
        'Outlier': String(length=12),
        'TotalGHGEmissions': Float,
        'GHGEmissionsIntensity': Float
    }

    df.to_sql(
        'buildings', 
        engine, method='multi', 
        if_exists="replace", 
        index=False,
        schema='public',
        dtype=types
        )
    
    print('The database has been loaded')
    
    with engine.connect() as conn:
        conn.execute(text("""ALTER TABLE buildings ADD PRIMARY KEY("OSEBuildingID");"""))
        conn.commit()
        print('Primary key has been added')

    
if __name__=='__main__':
    main()

