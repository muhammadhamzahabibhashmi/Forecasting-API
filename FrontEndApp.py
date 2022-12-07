import json
import requests
import os
import pandas as pd
import streamlit as st
try:
    os.mkdir('Stores/')
except:
    pass

st.title("Forecasting!")
genre = st.sidebar.radio(
    "Select one",
    ('Train', 'CheckStatus', 'Prediction'))

if genre == 'Train':
    uploaded_file = st.file_uploader("Choose a file",type=['csv'])
    if uploaded_file is not None:
        my_string = pd.read_csv(uploaded_file)
        NumberOfEpochs = st.number_input('Number Of Epochs to Train on', min_value=1, step=1)
        # st.write(uploaded_file.name)
        st.write('The Epochs are ', NumberOfEpochs)
        st.write(my_string)
        if st.sidebar.button('Train'):
            payload = (my_string.to_dict())
            payload2 = {"UserID":'245',"NumberOfEpochs":NumberOfEpochs,"FileName":str(uploaded_file.name)}
            payyl={"payload1":payload,"payload2":payload2}
            resspoon = requests.post(url="http://192.168.18.81:5000/train/", json=payyl)
            st.write('Go to CheckStatus')
            # try:
            #     st.write("Done Training")
            # except:
            #     st.write("Running")

elif genre == 'CheckStatus':
    if st.sidebar.button('Check Status Of Training'):
        payload2 = {"UserID":"245"}
        asd2=requests.post(url="http://192.168.18.81:5000/Checktrain/", json=payload2)
        st.write(json.loads(asd2.text))

elif genre == 'Prediction':
    payload2 = {"UserID":"245"}
    asd2=requests.post(url="http://192.168.18.81:5000/Checktrain/", json=payload2)
    ListOfUsersModelsmodelss = json.loads(asd2.text)['ListOfUsersModels']
    newListOfUsersModelsmodelss = []
    for drw in ListOfUsersModelsmodelss:
        if (json.loads(asd2.text)[drw]['Training']) == "Completed":
            newListOfUsersModelsmodelss.append(drw)
    if len(newListOfUsersModelsmodelss) > 0:
        optionforstores = st.sidebar.selectbox('Select a model for predictions',newListOfUsersModelsmodelss)
        ListOfUsersModels = json.loads(asd2.text)[optionforstores]['ListOfStores']
        optionforstore = st.sidebar.selectbox('Select a store for predictions',ListOfUsersModels)
        forecastDayysnumber = st.sidebar.number_input('Forecast Days', min_value=1, step=1)
        if st.sidebar.button('Predictions'):
            payyl = {"UserID":'245',"store":optionforstore,"days":forecastDayysnumber,"FileName":optionforstores}
            asd=requests.post(url="http://192.168.18.81:5000/detect/", json=payyl)
            json_load = json.loads(asd.text)
            st.write(f"Loading Predictions For {optionforstore}")
            st.write(json_load)
    else:
        st.write("No model available for prediction. Check status and wait if training is In Progress")
