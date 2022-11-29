#!/usr/bin/python
# coding: utf-8
import pickle
import pickle,os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from flask import Flask, request 

application = Flask(__name__)
try:
    os.mkdir('Models')
except:
    pass
try:
    os.mkdir('Processed_Dataset')
except:
    pass
try:
    os.mkdir('DataTransforms')
except:
    pass

global jsson 
jsson = {}


@application.route('/')
def entry_point():
    return "Forecasting Prices Of Stores API Status -->> Active "

@application.route('/detect/', methods=['POST'])
def detect():
    if request.method == 'POST':
        arggg = (request.json)
        store = arggg['store']
        days = arggg['days']
        UserID = arggg['UserID']
        model = load_model(f'Models/{UserID}_{store}.h5')
        df = pd.read_csv(f'Processed_Dataset/{UserID}_{store}.csv',index_col='Timestamp',parse_dates=True)
        with open(f'DataTransforms/{UserID}_{store}.pickle', 'rb') as handle:
                scaler = pickle.load(handle)
        scaled_train  = scaler.transform(df)
        test_predictions = []
        first_eval_batch = scaled_train[-12:]
        current_batch = first_eval_batch.reshape((1, 12, 1))
        for i in range(days):
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        data = scaler.inverse_transform(test_predictions)
        pred = []
        for ttrr in data:
            pred.append(round(ttrr[0],2))
        asd={}
        if (len(pred)>0):
            asd['Predictions'] = pred
            return asd
        else:
            return { 'message' :  "Unable to predict" }

@application.route('/Checktrain/', methods=['POST'])
def checktrain():
    global jsson
    if request.method == 'POST':
        arggg = request.json
        print(jsson)
        print(str(arggg['UserID']))
        return jsson[str(arggg['UserID'])]


@application.route('/train/', methods=['POST'])
def train():
    global jsson
    if request.method == 'POST':
        arggg = (request.json)
        FileDataframe = (arggg)
        asd=(FileDataframe["payload1"])
        dff=pd.DataFrame(asd)
        UserID = (FileDataframe["payload2"]['UserID'])
        jsson[UserID] = "False"

        new_dataframe = dff[["Store","Price","Timestamp"]]
        timee_day = []
        for things in new_dataframe["Timestamp"]:
            timee_day.append(things.split('T')[0])
        new_dataframe.drop("Timestamp", axis = 1, inplace = True)
        new_dataframe["Timestamp"]=timee_day
        new_dataframe.to_csv('Processed_Dataset/Processed.csv',index=False)        
        new_dataframee = pd.read_csv('Processed_Dataset/Processed.csv',index_col='Timestamp',parse_dates=True)
        stores = list(set(new_dataframee.Store.values.tolist()))
        asd={}
        rmsee = []
        for stro in stores:
            final_df = new_dataframee.loc[new_dataframee['Store'] == stro]

            df = final_df[["Price"]]
            splitter = int((4.5 * len(df)) / 5)
            train = df.iloc[:splitter]
            test = df.iloc[splitter:]
            test[-12:].to_csv(f"Processed_Dataset/{UserID}_{stro}.csv")
            scaler = MinMaxScaler()
            scaler.fit(train)
            with open(f'DataTransforms/{UserID}_{stro}.pickle', 'wb') as handle:
                pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
            scaled_train = scaler.transform(train)
            # scaled_test = scaler.transform(test)
            scaled_train[:10]
            n_input = 12
            generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
            model = Sequential()
            model.add(LSTM(100, return_sequences=True,activation='relu', input_shape=(n_input, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mse')
            model.summary()
            model.fit(generator,epochs=1)
            model.save(f'Models/{UserID}_{stro}.h5')

            test_predictions = []
            first_eval_batch = scaled_train[-n_input:]
            current_batch = first_eval_batch.reshape((1, n_input, 1))
            for i in range(len(test)):
                current_pred = model.predict(current_batch)[0]
                test_predictions.append(current_pred)
                current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
            del model
            test.head()
            true_predictions = scaler.inverse_transform(test_predictions)
            pred = []
            for ttrr in true_predictions:
                pred.append(round(ttrr[0],2))
            test.insert(loc=1, column="Predictions", value=pred)
            tttttt = mean_squared_error(test['Price'],test['Predictions'], squared=False)
            rmsee.append(tttttt)
            asd[f'{stro}'] = tttttt

        if (len(pred)>0):
            asd['AverageRMSE'] = round((sum(rmsee) / len(rmsee)),2)
            asd['Status'] = 1
            asd['Stores'] = stores
            jsson[UserID] = "True"

            return asd
        else:
            return { 'message' :  "Unable to predict" }


if __name__ == '__main__':
    application.run(host='0.0.0.0',threaded=True)
