# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:53:51 2022

@author: faruk
"""

from flask import Flask, jsonify, request
#import numpy as np

import pandas as pd
#import numpy as np

import Elo_Final as final
# https://www.tutorialspoint.com/flask

import flask
app = Flask(__name__)

@app.route('/')
def home_page():
    return flask.render_template('index.html')

@app.route('/predict',methods = ['POST'] )
def predict():
    print(" HI")
    card_id = request.form.to_dict()
    card_id = str(card_id['card_id'].strip())
    print(card_id)
    data = pd.DataFrame({'card_id' : [card_id],'feature_1':['']})
    
    prediction = final.function_1(data)
    #print("Loyalty Score", prediction)
    
    return jsonify({'prediction': str(prediction)})

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)