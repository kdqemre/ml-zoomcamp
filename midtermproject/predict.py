# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 06:04:21 2021

@author: Kdq
"""

import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = f'RFmodel.bin'

with open(model_file, 'rb') as f_in: 
    RFmodel = pickle.load(f_in)
    
RFmodel



app = Flask("water_pump")

@app.route("/predict", methods=["POST"])
def predict(pump):
    pump = request.get_json()
   
    
    y_pred = RFmodel.predict(pump)
    return y_pred

    result = {
        
        "pump_status": y_pred
            
        }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)