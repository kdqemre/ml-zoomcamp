# -*- coding: utf-8 -*-
"""
@author: Kdq
"""

import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = f'RFmodel.bin'

with open(model_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)






app = Flask("pump_status")

@app.route("/predict", methods=["POST"])
def predict():
    pump = request.get_json()
   
    X_pump = dv.transform([pump])
    y_pred = model.predict(X_pump)

    y_prob = (model.predict_proba(X_pump)).max()

    pump_status='' 

    if y_pred == 0:
        pump_status='non_functional'

    elif y_pred == 1:
        pump_status='functional'
    
    elif y_pred == 2:
        pump_status="functional_needs_repair"
        

    result = {
        
        "pump_status": pump_status,
        "probability": float(y_prob)
            
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)



