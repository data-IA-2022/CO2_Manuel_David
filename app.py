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
from interpret.glassbox import ExplainableBoostingRegressor, RegressionTree, LinearRegression
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
    

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/report")
def report():
    return render_template('report.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    data = {}
    if request.method=='POST':
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
        
        return render_template('prediction.html', results = results, method=request.method)
        
    return render_template('prediction.html', method = request.method)

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
