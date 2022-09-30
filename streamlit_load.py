# ------------------------------------
# import packages
# ------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_shap import st_shap
import shap
import json
import time
from pandas import json_normalize
import seaborn as sns
from shap.plots import waterfall
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle
from PIL import Image
import io
import zlib
import requests


model_threshold = 48.7

##############################################################################
#                         main function
##############################################################################
# ----------------------------------------------------
def main():
    # ------------------------------------------------
    # Configuration of the streamlit page
    # -----------------------------------------------
    st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.markdown("<h1 style='text-align: center; color: green;'>Loan application scoring dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'> PECHAIRE CHARLY - Data Scientist</h2>", unsafe_allow_html=True)

    #st.title('Loan application scoring dashboard')
    #st.subheader("NOM PRENOM")

    # Display the LOGO
    img = Image.open("LOGO.png")
    st.sidebar.image(img, width=250)

    # Display the loan image
    img = Image.open("loan.png")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(img, width=350)
    with col3:
        st.write(' ')

    ###############################################################################
    #                      LIST OF API REQUEST FUNCTIONS
    ###############################################################################
    # ------------------------------------------------
    API_URL = "https://projet7ocdashboard.herokuapp.com/api/" # ton URL

    # -----------------------------------------------
    '''
    @st.cache
    def fetch_cust_ids():
        # URL of the sk_id API
        list_cust_id_api_url = API_URL + 'list_cust_id/'
        cust_ids = requests.post(list_cust_id_api_url).json()
        cust_ids = cust_ids['ids']
        return cust_ids

    @st.cache
    def fetch_data_cust_by_id(selected_id):
        # URL of the sk_id API
        data_cust_by_id_api_url = API_URL + "data_cust_by_id/?id=" + str(selected_id)
        data_cust_by_id = requests.get(data_cust_by_id_api_url).json()
        data_cust_by_id = pd.DataFrame.from_dict(data_cust_by_id['data'])
        return data_cust_by_id

    #############################################################################
    #                          Selected id
    #############################################################################
    cust_ids = fetch_cust_ids()
    selected_id = st.sidebar.selectbox('Select a customer ID:', cust_ids)
    st.write('You have selected customer ID:', selected_id)

    ##############################################################################
    #                         Customer's data checkbox
    ##############################################################################
    data_cust_by_id = fetch_data_cust_by_id(selected_id)
    if st.sidebar.checkbox("Customer's Data"):
        st.subheader('Data of the selected customer')
        st.write(data_cust_by_id)
    '''
#    
    


if __name__ == "__main__":
    main()