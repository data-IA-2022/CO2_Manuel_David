# save this as app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/report")
def report():
    return render_template('report.html')

@app.route("/prediction")
def predict():
    return render_template('prediction.html')

