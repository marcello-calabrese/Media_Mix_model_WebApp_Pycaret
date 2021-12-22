from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# We make the layout of the app wide as the app is a single page
st.set_page_config(layout="wide")

# Load the trained model for predictions
model = load_model('Final ExtraTreeRegr_feateng_preproc Model2 20Dic2021')

# Create the Saturation or diminishing return function

@st.cache
def logistic_function(x_t, mu):
     '''
     param x_t: marketing spend vector (float)
     param mu: half-saturation point(float)
     return: transformed spend vector
     '''
     return (1 - np.exp(-mu * x_t)) / (1 + np.exp(-mu * x_t)) 

# Define the function to predict the model
@st.cache
def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df[['sales', 'Label']]
    #predictions = predictions_df['Label']
    
    return predictions


#################################################################################

# Building the wireframe of the APP

from PIL import Image
image = Image.open('What-is-Media-Mix-2.jpg')
image_main = Image.open('OldandNewMedia_1000px_resized.jpg')
st.sidebar.image(image, use_column_width=True)
st.sidebar.markdown('** Details about dataset, saturated spend and the model**')
st.sidebar.text(''' The dataset media spend variables 
are: 

- tv_sponsorships 
- tv_cricket
- tv_RON	
- radio
- NPP
- Magazines
- OOH
- Social
- Programmatic
- Display_Rest
- Search	
- Native''')
st.sidebar.markdown('** Saturation Effect or Diminishing Return**')
st.sidebar.markdown('''Increasing the amount of advertising 
increases the percent of the audience reached by the 
advertising, hence increases demand, but a 
linear increase in the advertising exposure doesn’t have 
a similar linear effect on demand.

**Typically** each incremental amount of 
advertising causes a progressively 
lesser effect on demand increase.
This is advertising saturation. 
**Usually Digital display ads 
and digital advertising in 
general have a high saturation effect, 
meanwhile TV, Radio have a low saturation effect.**''')

st.sidebar.markdown('''**Machine Learning Package**''' )

st.sidebar.markdown('''[PyCaret](https://pycaret.org/)''')

st.image(image_main, use_column_width='Never')
st.title('Media Mix Model Prediction')

st.text('''This app predicts sales on media mix spend for different media channels. 
Before the prediction, we apply a saturated spending function to the marketing spend vector.
For details about saturated spend function, dataset and machine learning package, check the side bar info on the left. 
The machine learning model used is the ExtraTreeRegression model.''')

# make a subheader  on streamlit
st.subheader('Upload the CSV file with the media spend data')

# upload a csv file on streamlit

file_upload = st.file_uploader("Upload a CSV file",  type=["csv"])

# if the file is uploaded, then show the dataframe
if file_upload is not None:
    data = pd.read_csv(file_upload, parse_dates=['Time'])
    st.dataframe(data.head())
    data_unseen_sat = pd.DataFrame({
    'tv_sponsorships_sat': logistic_function(data['tv_sponsorships'].values, 0.05),
    'tv_cricket_sat': logistic_function(data['tv_cricket'].values, 0.05),
    'tv_RON_sat': logistic_function(data['tv_RON'].values, 0.1),
    'radio_sat': logistic_function(data['radio'].values, 0.2),
    'NPP_sat': logistic_function(data['NPP'].values, 0.5),
    'Magazines_sat': logistic_function(data['Magazines'].values, 0.4),
    'OOH_sat': logistic_function(data['OOH'].values, 0.3),
    'Social_sat': logistic_function(data['Social'].values, 0.7),
    'Programmatic_sat': logistic_function(data['Programmatic'].values, 0.8),
    'Display_Rest_sat': logistic_function(data['Display_Rest'].values, 0.7),
    'Search_sat': logistic_function(data['Search'].values, 0.6),
    'Native_sat': logistic_function(data['Native'].values, 0.9),
    'sales': data['sales']
    
    })
    
    st.subheader('Data Frame with Unsaturated Media spend transformation. First 5 rows')
    st.dataframe(data_unseen_sat.head())
    
    # Predict the model from the unsaturated data
    st.subheader('Model Prediction')
    button = st.button('Start Predicion')
    
    if button:
        predictions = predict(model, data_unseen_sat)
        st.write(predictions)
        
        # create a line chart with the predicted sales and the actual sales
        st.subheader('Sales: Actual vs Predicted')
        st.line_chart(predictions)
    