import streamlit as st 
import numpy as np
import pandas as pd
import pickle 
from datetime import datetime
from datetime import date

model = pickle.load(open('ensemb.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
encoder = pickle.load(open('enc.pkl', 'rb'))
st.header("Surge Price Prediction App")
def predict():
    C1, C2 = st.columns(2)
    
    with C1:
        
        destination = st.selectbox("select destination ",['North Station', 'Haymarket Square', 'North_end','Beacon Hill', 'Boston University', 'West End', 'Berlin',
           'Financial District', 'Theatre District', 'South Station','Fenway','Northeastern University','Back Bay'],key='destination')
        timing=st.time_input('current time',value=None, step=60)
        hour = timing.hour
        minute = timing.minute
        current_date = date.today()
        selected_date = st.date_input("Select a date:", value=current_date)
        day=selected_date.weekday()
        

      
        
    with C2:
        
        wind = st.number_input("Current Wind")
        temperature  = st.slider("Current Temperature", min_value=30.00, max_value=60.00, value=30.00, step=0.01)
        pressure= st.slider("Current Pressure", min_value=700.00, max_value=1300.00, value=700.00, step=0.01)
        humidity= st.slider("Current Humidity", min_value=0.00, max_value=1.00, step=0.01)
        distance = st.number_input("Current Distance", max_value=8.00, step=0.01)
        rain= st.number_input("Current rain", min_value=0.0001,max_value=0.8000, format="%.4f", step=0.0001)
        source = st.selectbox("select location ",le.classes_,key='source')
        
        feat = np.array([distance,temperature,pressure,rain,humidity,wind,day,hour,minute,destination,source]).reshape(1,-1)
        cols= ['distance', 'temp', 'pressure', 'rain','humidity','wind','day', 'hour',
             'minute','destination','source']
        

        feat1 = pd.DataFrame(feat,columns=cols)
        return feat1

frame = predict()

def prepare(d):
    col2 = d.columns
    enc_data = pd.DataFrame(encoder.transform(d[['destination']]).toarray())
    enc_data.columns = encoder.get_feature_names_out()

    d = d.join(enc_data)

    d.drop(['destination'], axis=1,inplace=True)
    d['categorysource']=le.transform(d['source'])[0]
#     d['categorysource'] = le.fit_transform(d['source'])
    d.drop('source', axis=1,inplace=True)
    
#         d = pd.DataFrame(d, columns=col2)
#     col2 = d.columns
    
    
   
    return d



if st.button('Predict'):
    frame2= prepare(frame)
    frame2=frame2.astype(float)
#     pred = model.predict(frame2)
#     predb = modelrf.predict(frame2)
    prex=model.predict(frame2)
#     st.write(isinstance(frame2, np.ndarray))
#     st.write(frame2['temp'].dtype)
#     st.write(pred)
#     st.write(predb)
    st.write(prex)
#     st.write(frame2)
    
