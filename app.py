# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:24:34 2021

@author: Shivam Ghildiyal
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open('modle.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "This is the end."

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Predition is"+ str(prediction)


@app.route('/predict_file',methods=["POST"])
def prediction_test():

    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=model.predict(df_test)
    
    return str(list(prediction))

if __name__ == '__main__':
    app.run()