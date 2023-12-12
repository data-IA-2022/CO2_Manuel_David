# save this as app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import dotenv
from utils import get_engine, get_df_from_db, interpret_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    RegressionTree,
    LinearRegression
    )
from urllib.parse import urlsplit, urlunsplit

app = Flask(__name__)

dotenv.load_dotenv()


def format_interpret_url(url):
    parsed = list(urlsplit(url))
    parsed_host = parsed[1].split(':')
    parsed_host[0] = 'co2-app.azurewebsites.net'
    new_host = ':'.join(parsed_host)
    parsed[1] = new_host
    return urlunsplit(parsed)


def tf_num_to_bool(x):
    if x > 0:
        return True
    else:
        return False


def format_updload_data(df):
    bool_cols = ['SteamUse(kBtu)', 'NaturalGas(kBtu)']
    df[bool_cols] = df[bool_cols].apply(lambda x: x.apply(lambda x: True if x > 0 else False))
    return df


@app.route("/")
def index():
    return render_template('home.html')


@app.route("/report")
def report():
    return render_template('report.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    df = get_df_from_db(get_engine(echo_arg=True))
    primary_types = df['PrimaryPropertyType'].unique()
    building_types = df['BuildingType'].unique()
    data = {}
    if request.method == 'POST':
        data['LargestPropertyUseTypeGFA_log'] = np.log10(float(request.form['superficie']))
        data['PrimaryPropertyType'] = request.form['util']
        data['BuildingType'] = request.form['type']
        data['Have_NaturalGas_Energy'] = int(request.form['gaz'])
        data['Have_Stream_Energy'] = int(request.form['steam'])
        
        df = pd.DataFrame([data])
        
        model = joblib.load('models/mlp.pkl')
        prediction = model.predict(df)
        value1, value2 = prediction[0]
        co2, nrj = np.exp(2.303 * value1), np.exp(2.303 * value2)
        results = [np.round(co2, 2), np.round(nrj, 2)]
        
        return render_template(
            'prediction.html', 
            results = results, 
            method=request.method,
            primary_types=primary_types,
            buiding_types=building_types)
        
    return render_template(
        'prediction.html', 
        method=request.method,
        primary_types=primary_types,
        buiding_types=building_types)


@app.route('/model')
def learning_curve_display():
    path = 'static/img'
    fns = [os.path.join(path, fn) for fn in os.listdir(path)]
    print(fns)
    return render_template('learning.html', images=fns)


@app.route('/diapo')
def display_diapo():
    path = 'static/pdf/CO2-MD-diapo.pdf'
    return render_template('display_diapo.html', pdf=path)

@app.route('/upload', methods=['GET', 'POST'])
def display_upload():
    if request.method == 'POST':
        fn = request.files.get("upload", None)
        df = pd.read_csv(fn, sep=';')
        df_test = df[['BuildingType', 
                      'PrimaryPropertyType',
                      'LargestPropertyUseTypeGFA', 
                      'SteamUse(kBtu)',
                      'NaturalGas(kBtu)',
                      'SiteEnergyUse(kBtu)', 
                      'TotalGHGEmissions']]
        df_test = format_updload_data(df_test)
        df_test = df_test.rename(columns={
            'SteamUse(kBtu)': 'Have_Stream_Energy',
            'NaturalGas(kBtu)': 'Have_NaturalGas_Energy',
            'SiteEnergyUse(kBtu)': 'SiteEnergyUse_kBtu_',
        })
        df_test['LargestPropertyUseTypeGFA_log'] = df_test['LargestPropertyUseTypeGFA'].apply(lambda x: np.log10(x))
        X = df_test[[
            'BuildingType',
            'PrimaryPropertyType',
            'LargestPropertyUseTypeGFA_log', 
            'Have_Stream_Energy',
            'Have_NaturalGas_Energy'
        ]]
        
        model = joblib.load('models/mlp.pkl')
        predictions = model.predict(X)
        co2, nrj = [], []
        for prediction in predictions:
            value1, value2 = np.exp(2.303 * prediction[0]), np.exp(2.303 * prediction[1])
            co2.append(value1)
            nrj.append(value2)
        df_test['Emissions_predicted'] = co2
        df_test['Consommation_predicted'] = nrj
        return render_template(
            'upload.html', 
            tables=[df_test.to_html(classes='data')], 
            titles=df_test.columns.values,
            method=request.method
        )
    return render_template('upload.html', method=request.method)


@app.route('/interpret', methods=['GET', 'POST'])
def display_interpret():
    engine = get_engine(echo_arg=True)
    df = get_df_from_db(engine)
    y1, y2 = df['TotalGHGEmissions_log'], df['SiteEnergyUse_kBtu_log']
    X_cols = ['Have_Stream_Energy', 'Have_NaturalGas_Energy', 'PrimaryPropertyType', 
              'LargestPropertyUseTypeGFA_log']
    X = df[X_cols]
    
    ebm = ExplainableBoostingRegressor()
    ebm2 = RegressionTree()
    ebm3 = LinearRegression()
    X_cat = X.select_dtypes(include=[object, bool])
    X_num = X.select_dtypes(exclude=[object, bool])
    
    preparation = ColumnTransformer(transformers=[
        ('tf_cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), X_cat.columns),
        ('tf_num', RobustScaler(), X_num.columns)
    ])
    
    models = [ebm, ebm2, ebm3]
    model_names = ['GradientBoosting', 'DesicionTree', 'LinearRegression']
    target_names = ['Emissions', 'Consommation']
    
    if request.method == 'POST':
        y = y1 if request.form['targets'] == 'Emissions' else y2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        
        X_tf_train = preparation.fit_transform(X_train)
        X_tf_test = preparation.fit_transform(X_test)
        names = preparation.get_feature_names_out(X_cols)

        df_train = pd.DataFrame(data=X_tf_train, columns=names)
        df_test = pd.DataFrame(data=X_tf_test, columns=names)

        for model, name in zip(models, model_names):
            if name == request.form['models']:
                port = np.random.randint(7000, 8000)
                print(port)
                url = interpret_model(model, names, df_train, y_train, df_test, y_test, port)
                return render_template(
                    'interpretml.html', 
                    options=model_names, 
                    targets=target_names, 
                    method=request.method,
                    plot_iml=url
                    )
    return render_template(
        'interpretml.html', 
        options=model_names, 
        targets=target_names,
        method=request.method)
