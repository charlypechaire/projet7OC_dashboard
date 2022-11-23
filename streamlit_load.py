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
from os.path import join, dirname
import os

model_threshold = 48.7





##############################################################################
# main function
##############################################################################
# ----------------------------------------------------
def main():
    
    if not os.environ.get("APP_ENV"):
        from os.path import join, dirname
        from dotenv import load_dotenv
        dotenv_path = join(dirname(__file__), '.env')
        load_dotenv(dotenv_path)
    # ------------------------------------------------
    # Configuration of the streamlit page
    # -----------------------------------------------
    st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.markdown("<h1 style='text-align: center; color: green;'>Loan application scoring dashboard</h1>", unsafe_allow_html=True)

    #st.title('Loan application scoring dashboard')

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
    # LIST OF API REQUEST FUNCTIONS
    ###############################################################################
    # ------------------------------------------------
    API_URL = os.environ.get("API_URL")
    
    # -----------------------------------------------
    def fetch_cust_ids():
        # URL of the sk_id API
        list_cust_id_api_url = API_URL + "list_cust_id/"
        cust_ids = requests.post(list_cust_id_api_url).json()
        cust_ids = cust_ids["ids"]
        return cust_ids
    
    def fetch_data_cust_by_id(selected_id):
        # URL of the sk_id API
        data_cust_by_id_api_url = API_URL + "data_cust_by_id/?id=" + str(selected_id)
        data_cust_by_id = requests.get(data_cust_by_id_api_url).json()
        data_cust_by_id = pd.DataFrame.from_dict(data_cust_by_id['data'])
        return data_cust_by_id

    def fetch_score_cust_by_id(selected_id):
        # URL of the sk_id API
        score_api_url = API_URL + 'score_by_id/?id=' + str(selected_id)
        score_value = requests.get(score_api_url).json()
        score_value = score_value['score']
        return score_value

    def fetch_data_of_all_cust():
        # URL of the sk_id API
        X_test_api_url = API_URL + "X_test/"
        X_test_dict = json.loads(requests.post(X_test_api_url).json())
        X_test = pd.DataFrame(X_test_dict)
        return X_test

    def fetch_feat_desc():
        # URL of the sk_id API
        df_features_desc_api_url = API_URL + "df_features_desc/"
        df_features_desc_dict = json.loads(requests.get(df_features_desc_api_url).json())
        df_features_desc = pd.DataFrame(df_features_desc_dict)
        return df_features_desc


    def fetch_explainer_expected_value():
        # URL of the sk_id API
        explainer_expected_value_api_url = API_URL + 'explainer_expected_value/'
        explainer_expected_value = requests.get(explainer_expected_value_api_url).json()
        explainer_expected_value = explainer_expected_value['explainer_expected_value']
        return explainer_expected_value

    
    def fetch_shap_values():
        # URL of the sk_id API
        shap_values_api_url = API_URL + "shap_values/"
        shap_values = requests.get(shap_values_api_url)
        decodedArrays = json.loads(shap_values.content)
        finalNumpyArray = np.asarray(decodedArrays["array"])
        return finalNumpyArray


    def fetch_shap_value_by_id(selected_id):
        # URL of the sk_id API
        shap_by_id_api_url = API_URL + 'shap_value_by_id/?id=' + str(selected_id)
        shap_value_by_id = requests.get(shap_by_id_api_url).json()
        shap_value_by_id = pd.DataFrame.from_dict(shap_value_by_id['data'])
        shap_value_by_id = shap_value_by_id.to_numpy()
        return shap_value_by_id

    ############################################################################
    # Graphics Functions
    ############################################################################
    # Global SHAP SUMMARY
    def display_shap_summary(nb_features, plot_type=None):
        shap_values = fetch_shap_values()
        X_test = fetch_data_of_all_cust()
        X_test_without_score = X_test.drop(['score', 'classes', 'index', 'TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
        if plot_type:
            shap.summary_plot(shap_values, X_test_without_score, max_display=nb_features, plot_type="bar")
        else:
            shap.summary_plot(shap_values, X_test_without_score, max_display=nb_features)
        plt.gcf()
        st.pyplot(plt.gcf())

    # Local SHAP Graphs
    def force_plot(selected_id):
        X_test = fetch_data_of_all_cust()
        data = X_test.drop(['score', 'classes', 'index', 'TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
        explainer_expected_value = fetch_explainer_expected_value()
        shap_value_by_id = fetch_shap_value_by_id(selected_id)
        return shap.force_plot(explainer_expected_value, shap_value_by_id, data.columns)

    # Gauge chart
    def gauge_chart(score_value):
        fig = go.Figure(go.Indicator(mode="gauge+number",
                                     number={'suffix': "%", 'font': {'size': 50}},
                                     value=float(score_value),
                                     title={'text': "Selected Customer Score", 'font': {'size': 30}},
                                     domain={'x': [0, 1], 'y': [0, 1]},
                                     gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                            'bar': {'color': "darkblue"},
                                            'bgcolor': "white",
                                            'borderwidth': 2,
                                            'bordercolor': "gray",
                                            'steps': [{'range': [0, 33], 'color': 'green'},
                                                      {'range': [33, 66], 'color': 'yellow'},
                                                      {'range': [66, 100], 'color': 'red'}]}))

        fig.update_layout(font={'color': "darkblue", 'family': "Arial"},
                          paper_bgcolor="lavender",
                          xaxis={'showgrid': False, 'showticklabels': False, 'range': [-1, 1]},
                          yaxis={'showgrid': False, 'showticklabels': False, 'range': [0, 1]},
                          plot_bgcolor='rgba(0,0,0,0)')

        return fig

    #############################################################################
    # Selected id
    #############################################################################
    cust_ids = fetch_cust_ids()
    selected_id = st.sidebar.selectbox('Select a customer ID:', cust_ids)
    st.write('You have selected customer ID:', selected_id)
    
    ##############################################################################
    # Customer's data checkbox
    ##############################################################################
    data_cust_by_id = fetch_data_cust_by_id(selected_id)
    if st.sidebar.checkbox("Customer's Data"):
        st.subheader('Data of the selected customer')
        st.write(data_cust_by_id)

    ##############################################################################
    # Model's decision checkbox
    ##############################################################################
    if st.sidebar.checkbox("Model's Decision"):
        st.subheader("Model's Decision")
        # Display score (probability):
        score_value = fetch_score_cust_by_id(selected_id)
        st.write('Customer score (probability): {:.1f}%'.format(score_value))
        # Display default threshold
        st.write(f'Model threshold: {model_threshold}%')
        # Compute decision according to the best threshold
        if score_value <= model_threshold:
            decision = "Loan granted"
        else:
            decision = "Loan rejected"
        st.write("Decision :", decision)

        ##########################################################################
        # Display customer's gauge meter chart (checkbox)
        ##########################################################################
        figure = gauge_chart(score_value)
        st.plotly_chart(figure)
        # Add markdown
        st.markdown('_Gauge meter plot for the applicant customer_')
        expander = st.expander("Concerning the classification model...")
        expander.write("The prediction was made using the XGBoost Classifier Model")
        expander.write("The default model is calculated to maximize air under ROC curve => maximize \
                        True Positives rate (TP) detection and minimize False Negatives rate (FP)")
    ##########################################################################
    # Display local SHAP force plot checkbox
    ##########################################################################
    if st.checkbox('Display force plot interpretation', key=25):
        shap_force_plot = force_plot(selected_id)
        st_shap(shap_force_plot, height=150, width=800)
        # Add markdown
        st.markdown('_SHAP Force Plot for the applicant customer_')
        expander = st.expander("Concerning the SHAP force plot...")
        expander.write("The above force plot allows you to see how features contributed \
                        to the individual prediction of the applicant customer. The bold value \
                        is the model’s score for this observation. Higher scores lead the model \
                        to predict 1 (loan rejected) and lower scores lead the model to predict 0 (loan granted). \
                        The features \ that were important to making the prediction for this applicant customer \
                        are shown in red and blue, with red representing features that pushed the \
                        model score higher, and blue representing features that pushed the score lower. \
                        Features that had more of an impact on the score are located closer to the \
                        dividing boundary between red and blue, and the size of that impact is represented by the size of the bar.")

    ##############################################################################
    # Importance of the features checkbox
    ##############################################################################
    if st.sidebar.checkbox("Importance Of The Features"):
        st.subheader('Importance Of The Features')
        nb_features = st.slider("Number of features to display",
                                min_value=2,
                                max_value=50,
                                value=10,
                                step=None,
                                format=None,
                                key=14)
        if st.checkbox('plot_type="bar"', key=30):
            display_shap_summary(nb_features, plot_type="bar")
        else:
            display_shap_summary(nb_features)

    ##########################################################################
    # Display Bi-variate analysis checkbox
    ##########################################################################
    if st.sidebar.checkbox("Display Bi-variate Analysis"):
        st.subheader('Display Bi-variate Analysis')
        X_test = fetch_data_of_all_cust()
        selected_row = X_test[X_test['SK_ID_CURR'] == selected_id].iloc[0]
        shap_values = fetch_shap_values()
        f_import = list(X_test.columns[np.argsort(np.abs(shap_values).mean(0))])
        f_import.remove('SK_ID_CURR')
        list_num_features = list(X_test.select_dtypes(include='number').columns)
        features = [f for f in f_import if f in list_num_features]

        st.markdown('Select features to display:')
        col1, col2 = st.columns(2)
        feature1 = col1.selectbox('Feature 1', reversed(features))
        feature2 = col2.selectbox('Feature 2', reversed(features))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        palette = cm.get_cmap('RdYlGn')
        ax.scatter(X_test[feature1], X_test[feature2], s=20, c=X_test['score'], cmap=palette.reversed())

        score_value = fetch_score_cust_by_id(selected_id)
        # color = 'green' if score_value <= model_threshold else 'red'
        cmap = cm.get_cmap(palette).reversed()
        color = cmap(score_value / 100)
        ax.plot(selected_row[feature1], selected_row[feature2], '*', ms=15, markeredgecolor='darkblue', color=color)
        ax.text(selected_row[feature1] - (X_test[feature1].max() * 0.06),
        selected_row[feature2] - (X_test[feature2].max() * 0.05), 'Selected id', color='darkblue', fontsize=12)

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        st.pyplot(fig)

        expander = st.expander("Concerning the scatterplot graph...")
        expander.write("This scatterplot graph shows the relationship between the two selected features, \
                       with a color gradient according to the score of the customers, and the positioning of the customer.")

    ##########################################################################
    # Display the feature distribution by classes
    ##########################################################################
    if st.sidebar.checkbox("Display Feature Distribution by Classes"):
        st.subheader('Display Feature Distribution by Classes')

        X_test = fetch_data_of_all_cust()
        shap_values = fetch_shap_values()
        f_import = list(X_test.columns[np.argsort(np.abs(shap_values).mean(0))])
        f_import.remove('SK_ID_CURR')
        list_num_features = list(X_test.select_dtypes(include='number').columns)
        features = [f for f in f_import if f in list_num_features]
        selected_feature = st.selectbox('Select feature to display:', reversed(features))

        X_test['classes'] = ['accepted' if x <= model_threshold/100 else 'rejected' for x in X_test['score']]
        classes = X_test['classes'].unique()
        groupes = []
        for cat in classes:
            groupes.append(X_test[X_test['classes'] == cat][selected_feature])

        plt.figure(figsize=(8, 6))

        medianprops = {'color': "black"}
        meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

        sns.boxplot(x='classes', y=selected_feature, data=X_test, showfliers=True, showmeans=True,
                    medianprops=medianprops, meanprops=meanprops, palette=['green', 'red'])

        selected_row = X_test[X_test['SK_ID_CURR'] == selected_id].iloc[0]
        score_value = fetch_score_cust_by_id(selected_id)
        #color = 'green' if score_value >= model_threshold else 'red'
        if selected_row['classes'] == 'accepted':
            plt.plot(0, selected_row[selected_feature], '*', markersize=15, color='darkblue')
            plt.text(0, selected_row[selected_feature], 'Selected id', color='darkblue', fontsize=12)
        else:
            plt.plot(1, selected_row[selected_feature], '*', markersize=15, color='darkblue')
            plt.text(1, selected_row[selected_feature], 'Selected id', color='darkblue', fontsize=12)

        plt.xlabel('')
        plt.xticks(size=12)
        plt.yticks(size=12)

        plt.gcf()
        st.pyplot(plt.gcf())

        expander = st.expander("Concerning the boxplot graph...")
        expander.write("This boxplot graph shows the the feature distribution by classes \
                        show the dispersion of the preprocessed features values used by the model to make a prediction. \
                        The green boxplot are for the customers that are accepted for their loan, and red boxplots are \
                        for the customers that are rejected. Values for the applicant customer are superimposed in blue star marker.")



if __name__ == "__main__":
    main()
