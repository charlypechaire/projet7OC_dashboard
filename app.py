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
from dotenv import load_dotenv
from os.path import join, dirname


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


app = Flask(__name__)
API_URL = os.environ.get("API_URL")
@app.route('/', methods=['GET','POST'])
def test():
    print("ok3")
    return ('test.html')
'''
df_features_desc = pd.read_csv('data_P7.csv', low_memory=False)
df_features_desc.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)

X_test = pd.read_csv('test_P7.csv', low_memory=False)
X_test.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)
X_test['SK_ID_CURR'] = X_test['SK_ID_CURR'].astype('int')

df_shap_values = pd.read_csv('data_shap.csv')
df_shap_values.set_index('Unnamed: 0', inplace=True)

np_shap_values = np.load('np_shap.npy')

#load saved model
xgb = joblib.load('modele_xgb_P7.sav')

@app.route("/")
def hello():
    return render_template('home.html')

# Get selected customer's score
@app.route('/api/score_by_id/')
def get_score_selected_cust():
    # URL of the sk_id API
    # score_api_url = API_URL + "score_by_id/?id=" + str(selected_id)
    id = request.args.get('id', default=1, type=int)
    print(f"The selected id is: {id}")
    score = round(X_test[X_test['SK_ID_CURR'] == int(id)]['score'].values[0]*100,1)
    print(f'Found score: {score}')
    return {'score': score}

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
    list_cust_id = X_test["SK_ID_CURR"].tolist()
    return jsonify({"ids": list_cust_id})

@app.route('/api/explainer_expected_value/')
# Get the explainer.expected_value
def get_explainer_expected_value():
    # URL of the sk_id API
    # explainer_expected_value_api_url = API_URL + "explainer_expected_value/"
    with open('explainer_expected_value.json', 'r') as fp:
        explainer_expected_value = json.load(fp)
    explainer_expected_value = explainer_expected_value['explainer_expected_value']
    return {'explainer_expected_value' : explainer_expected_value}

@app.route('/app/get_shap/')
# find 20 nearest neighbors among the testing set
def get_shap():
    # URL of the sk_id API
    # neigh_cust_api_url = API_URL + "get_shap/?id=" + str(selected_id)
    # fit nearest neighbors among the selection
    NN = NearestNeighbors(n_neighbors=20)
    NN.fit(X_test)
    id = request.args.get('id', default=1, type=int)
    X_cust = X_test[X_test['SK_ID_CURR'] == int(id)].iloc[0]
    idx = NN.kneighbors(X=X_cust,
    n_neighbors=20,
    return_distance=False).ravel()
    nearest_cust_idx = list(X_test.iloc[idx].index)
    # data and target of neighbors
    # ----------------------------
    X_neigh = X_test.loc[nearest_cust_idx, :]
    # prepare the shap values of nearest neighbors + customer
    shap.initjs()
    # creating the TreeExplainer with our model as argument
    explainer = shap.TreeExplainer(xgb)
    # Expected values
    expected_vals = pd.Series(list(explainer.expected_value))
    # calculating the shap values of selected customer
    shap_vals_cust = pd.Series(list(explainer.shap_values(X_cust)[1]))
    # calculating the shap values of neighbors
    shap_val_neigh_ = pd.Series(list(explainer.shap_values(X_neigh)[1]))
    # Converting the pd.Series to JSON
    X_neigh_json = json.loads(X_neigh.to_json())
    expected_vals_json = json.loads(expected_vals.to_json())
    shap_val_neigh_json = json.loads(shap_val_neigh_.to_json())
    shap_vals_cust_json = json.loads(shap_vals_cust.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'X_neigh_': X_neigh_json,
                    'shap_val_neigh': shap_val_neigh_json, # double liste
                    'expected_vals': expected_vals_json, # liste
                    'shap_val_cust': shap_vals_cust_json}) # double liste

@app.route('/api/shap_values/', methods=['GET', 'POST'])
# Get the shap_values
def get_shap_values():
    global df_shap_values
    # URL of the sk_id API
    # shap_values_api_url = API_URL + "shap_values/"
    # Serialization
    numpyData = {"array": np_shap_values}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return encodedNumpyData

@app.route('/api/shap_value_by_id/')
# Get the shap_value_by_id
def get_shap_value_by_id():
    global df_shap_values
    # URL of the sk_id API
    # shap_by_id_api_url = API_URL + "shap_value_by_id/?id=" + str(selected_id)
    id = request.args.get('id', default=1, type=int)
    idx = X_test.index[X_test['SK_ID_CURR'] == int(id)][0]
    shap_value_by_id = df_shap_values.loc[idx].to_dict()
    return {'data': [shap_value_by_id]}

@app.route('/api/X_test/', methods=['GET', 'POST'])
# Get X_test
# URL of the sk_id API
# X_test_api_url = API_URL + "X_test/
def get_X_test():
    global X_test
    return jsonify(X_test.to_json())

@app.route('/api/df_features_desc/', methods=['GET'])
# Get df_features_desc
# URL of the sk_id API
# df_features_desc_api_url = API_URL + "df_features_desc/
def get_features_desc():
    global df_features_desc
    return jsonify(df_features_desc.to_json())
'''
if __name__ == "__main__":
    app.run(debug=True)
