from flask import Flask, render_template, request, make_response
from dotenv import load_dotenv
from app import stock_analysis
import json
import time
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    user = "Enrico"
    return render_template('index.html', name=user)

@app.route('/ic')
def image_classifier():
    return render_template('image_classifier.html')

@app.route('/st')
def stock_prediction():

    #TODO: Get the list of positive, neutral and negative tweets

    return render_template('stock_prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    days = request.form['days'];
    stock = "BTC-USD"

    # TODO: check if days are numbers!
    maxConfidenceItem =  stock_analysis.prediction_data(stock, days)
    # maxConfidenceItem = stock_analysis.fake_predict(stock, days)
    # time.sleep(2)
    # print(maxConfidenceItem)
    maxConfidenceItem['prediction'] = '{} $'.format(str(maxConfidenceItem['prediction']))
    maxConfidenceItem['confidence'] = '{}%'.format(str(maxConfidenceItem['confidence']))

    return json.dumps({'status':'OK','result':maxConfidenceItem});

