# save this as app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import dotenv


app = Flask(__name__)

dotenv.load_dotenv()

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

