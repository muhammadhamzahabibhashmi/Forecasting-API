import json
from flask import jsonify
import requests
import os
import pandas as pd
import streamlit as st
try:
    os.mkdir('Stores/')
except:
    pass

st.title("Forecasting!")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    my_string = pd.read_csv(uploaded_file)
    st.write(my_string)
    if st.sidebar.button('Train'):
        payload = (my_string.to_dict())
        payload2 = {"UserID":'245'}
        payyl={"payload1":payload,"payload2":payload2}
        asd=requests.post(url="http://192.168.18.81:5000/train/", json=payyl)
        json_load = json.loads(asd.text)
        with open(r'Stores/stores.txt', 'w') as fp:
            for item in json_load["Stores"]:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')
        st.json(json_load)


ListOfStores = []
try:
    with open('Stores/stores.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            ListOfStores.append(x)


    optionforstore = st.sidebar.selectbox('Select a store for predictions',ListOfStores)
    forecastDayysnumber = st.sidebar.number_input('Forecast Days', min_value=1, step=1)

    if st.sidebar.button('Predictions'):
        payyl = {"UserID":'245',"store":optionforstore,"days":forecastDayysnumber}
        asd=requests.post(url="http://192.168.18.81:5000/detect/", json=payyl)
        json_load = json.loads(asd.text)
        st.write(f"Loading Predictions For {optionforstore}")
        st.write(json_load)
except:
    st.write("Train the model first")

