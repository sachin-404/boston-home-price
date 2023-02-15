from flask import Flask, request, jsonify, url_for, render_template
import pickle
import json
import numpy as np
import pandas as pd


app = Flask(__name__)

# load the pickle model
model = pickle.load(open('model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data) # this data will be in key value pair
    print(np.array(list(data.values())).reshape(1, -1))
    # transform the data for prediction
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
