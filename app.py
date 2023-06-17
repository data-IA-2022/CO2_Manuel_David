# save this as app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

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
        data['LargestPropertyUseTypeGFA_log'] = np.log(float(request.form['superficie']))
        data['PrimaryPropertyType'] = request.form['type']
        data['NumberofBuildings'] = int(request.form['nb_build'])
        data['Have_NaturalGas_Energy'] = int(request.form['gaz'])
        data['Have_Stream_Energy'] = int(request.form['steam'])
        
        df = pd.DataFrame([data])
        model = joblib.load('model.pkl')
        prediction = model.predict(df)
        value1, value2 = prediction[0]
        co2, nrj = np.exp(value1), np.exp(value2)
        results = [np.round(co2, 2), np.roud(nrj, 2)]
        
        return render_template('prediction.html', results = results, method=request.method)
        
    return render_template('prediction.html', method = request.method)

