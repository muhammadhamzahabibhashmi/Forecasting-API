#!/usr/bin/python
# coding: utf-8
import pickle
import pickle,os
import numpy as np
import pandas as pd
# from flask import jsonify
import json
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
FileNameee = {}
try:
    with open('history.json', 'r') as openfile:
        jsson = json.load(openfile)
except:
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
        FileName = arggg['FileName']
        model = load_model(f'Models/{UserID}_{FileName}_{store}.h5')
        df = pd.read_csv(f'Processed_Dataset/{UserID}_{FileName}_{store}.csv',index_col='Timestamp',parse_dates=True)
        with open(f'DataTransforms/{UserID}_{FileName}_{store}.pickle', 'rb') as handle:
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
        try:
            return json.dumps(jsson[str(arggg['UserID'])])
        except KeyError:
            jsson[str(arggg['UserID'])] = {"status": "False", "StoreForTraining":'This User is never been Trained'}
            return json.dumps(jsson[str(arggg['UserID'])]) 

@application.route('/train/', methods=['POST'])
def train():
    global jsson
    if request.method == 'POST':
        arggg = (request.json)
        FileDataframe = (arggg)
        asd=(FileDataframe["payload1"])
        dff=pd.DataFrame(asd)
        UserID = (FileDataframe["payload2"]['UserID'])
        NumberOfEpochs = (FileDataframe["payload2"]['NumberOfEpochs'])
        FileName = (FileDataframe["payload2"]['FileName'])
        try:
            if FileName.split('.')[0] not in FileNameee[UserID]: 
                FileNameee[UserID].append(FileName.split('.')[0])
        except KeyError:
            FileNameee[UserID] = []
            jsson[UserID] = {}
            FileNameee[UserID].append(FileName.split('.')[0])

        # FileNameee[UserID] = FileNameee[UserID]
        # UserID_FileName = []        
        # jsson[UserID] = f"{UserID} has received the file and started tr"
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
        donestores = []
        totallStores = len(stores) 
        IndextotallStores = 0
        for stro in stores:
            jsson[UserID]["ListOfUsersModels"] = FileNameee[UserID]  
            jsson[UserID][f"{FileName.split('.')[0]}"] = {"Training" :"In Progress" ,
            "ListOfStores":stores, "StoresTrainingStatus": f"{IndextotallStores} Out Of {totallStores} stores have been Trained"}
            
            final_df = new_dataframee.loc[new_dataframee['Store'] == stro]
            donestores.append(stro)
            df = final_df[["Price"]]
            splitter = int((4.5 * len(df)) / 5)
            train = df.iloc[:splitter]
            test = df.iloc[splitter:]
            test[-12:].to_csv(f"Processed_Dataset/{UserID}_{FileName.split('.')[0]}_{stro}.csv")
            scaler = MinMaxScaler()
            scaler.fit(train)
            with open(f"DataTransforms/{UserID}_{FileName.split('.')[0]}_{stro}.pickle", 'wb') as handle:
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
            model.add(LSTM(units=196,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=196,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=196,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=96,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=36))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mse')
            model.summary()
            model.fit(generator,epochs=NumberOfEpochs)
            model.save(f'Models/{UserID}_{FileName.split(".")[0]}_{stro}.h5')

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
            if (max(test['Price']) - min(test['Price'])) > 1:
                normlizedRMSEE = tttttt / (max(test['Price']) - min(test['Price'])) 
                if normlizedRMSEE > 1:
                    normlizedRMSEE = 1                
                rmsee.append(normlizedRMSEE)
                asd[f'{stro}'] = normlizedRMSEE
            elif (max(test['Price']) - min(test['Price'])) < 1:
                normlizedRMSEE = tttttt / 1
                if normlizedRMSEE > 1:
                    normlizedRMSEE = 1      
                rmsee.append( normlizedRMSEE)
                asd[f'{stro}'] = normlizedRMSEE
            IndextotallStores += 1
        jsson[UserID][f"{FileName.split('.')[0]}"] = {"ListOfStores":stores, "RMSE_StoreByStore": asd,
        "Training" :"Completed",
        "StoresTrainingStatus": f"{IndextotallStores} Out Of {totallStores} stores have been Trained"}    
        jsson[UserID][FileName.split('.')[0]]['NormalizedRMSE'] = round((sum(rmsee) / len(rmsee)),2)
        json_object = json.dumps(jsson, indent=4)
        with open("history.json", "w") as outfile:
            outfile.write(json_object)
        return 0
        

if __name__ == '__main__':
    application.run(host='0.0.0.0',threaded=True)
