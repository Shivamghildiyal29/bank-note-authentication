# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:24:34 2021

@author: Shivam Ghildiyal
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('modle.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "This is the api to connect the machine learning model with data."

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Predition is"+ str(prediction)


@app.route('/predict_file',methods=["POST"])
def prediction_test():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=model.predict(df_test)
    
    return str(list(prediction))

if __name__ == '__main__':
    app.run()