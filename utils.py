from sodapy import Socrata
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from interpret import perf, show, show_link


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

    url = os.environ['POSTGRES_URL']

    engine = create_engine(url, echo=echo_arg)
    return engine

def get_df_from_db(engine):
    with engine.begin() as conn:
        query = text("""SELECT * FROM "CO2_selected_colums_for_lm_no_outlier_prepared_6_colums"; """)
        df = pd.read_sql(query, conn)
    return df

def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('R2-score')
    plt.grid(ls='--')
    plt.legend(loc='best')
    fn = f'static/img/{title}.png'
    plt.savefig(fn)
    plt.show()

def interpret_model(model, X_cols, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    ebm_global = model.explain_global()
    ebm_local = model.explain_local(X_test, y_test)
    regression_perf = perf.RegressionPerf(model.predict, feature_names=X_cols)
    regression_explanation = regression_perf.explain_perf(X_test, y_test)
    # show([ebm_global, ebm_local, regression_explanation])
    return show_link([ebm_global, ebm_local, regression_explanation])

