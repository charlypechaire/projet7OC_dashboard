# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, Response
import json
import requests
import pandas as pd
from flask import request
import zlib
import io
import os
import numpy as np
from json import JSONEncoder
import numpy
import joblib
from sklearn.neighbors import NearestNeighbors
import shap
import pickle

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


app = Flask(__name__)
API_URL = "https://projet7ocdashboard.herokuapp.com/api/"

df_features_desc = pd.read_csv('data_P7.csv', low_memory=False)
df_features_desc.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)

X_test = pd.read_csv('test_P7.csv', low_memory=False)
X_test.drop('Unnamed: 0.1', axis=1, errors='ignore', inplace=True)
X_test.set_index('Unnamed: 0', inplace=True)
X_test['SK_ID_CURR'] = X_test['SK_ID_CURR'].astype('int')

#df_shap_values = pd.read_csv('df_shap_values.csv')
#df_shap_values.set_index('Unnamed: 0', inplace=True)

#np_shap_values = np.load('df_shap_values.npy')


#load saved model
balanced = joblib.load('modele_P7.sav')


@app.route("/")
def hello():
    return render_template('home.html')
    
# Get the customer's data by id
@app.route('/api/data_cust_by_id/')
def get_cust_data_by_id():
    # URL of the sk_id API
    # data_cust_by_id_api_url = API_URL + "data_cust_by_id/?id=" + str(selected_id)
    id = request.args.get('id', default=1, type=int)
    data_by_id = X_test[X_test['SK_ID_CURR'] == int(id)].iloc[0].to_dict()
    return {'data': [data_by_id]}

# Get the list of customer's id
@app.route('/api/list_cust_id/', methods=['GET','POST'])
def get_list_cust_id():
    # URL of the sk_id API
    # list_cust_id_api_url = API_URL + "list_cust_id/"
    list_cust_id = list(X_test['SK_ID_CURR'])
    return jsonify({'ids': list_cust_id})


if __name__ == "__main__":
    app.run(debug=True)