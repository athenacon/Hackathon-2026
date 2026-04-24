from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import *
import matplotlib.pyplot as plt


app = Flask(__name__)

# This route "/" is the default page, this will run when the browerser runs the URL
@app.route("/")
def index():
    return render_template("index.html")  # Pass the name variable to the template
    
# This is called by a acript in the index.html
@app.route("/run")

#------------------------------------------Pre-Processing-------------------------------------
def run_model():
    def pre_process(data, look_back = 20):
        #data = data.to_numpy()
        X = []
        y = []

        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i, 0])
            y.append(data[i, 0])

        return np.array(X), np.array(y)   
#----------------------------------------Read Data in----------------------------------------
    df = pd.read_csv("combined_data.csv")
    df1 = df.where(df['Area']=="Australia")
    df1 = df1.where(df['Crop']=="Maize")
    df1 = df1.dropna(how='any',axis=0) 
    df1['Year'] = pd.to_datetime(df1['Year'])
    df1.set_index("Year", inplace=True)
    data = df1["Crop_Yield"].astype(float).values.reshape(-1,1)
    #--------------------------------------------manipulate data------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    look_back = 20
    X, y = pre_process(data)

    target_dates = df.index[look_back:len(data)]
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, target_dates, test_size=0.3, shuffle=False)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    #---------------------------------------------Model-------------------------------------------
    model = Sequential()
    model.add(LSTM(16, input_shape=(1,1)))
    model.add(Dense(8))
    model.add(Dense(1, 'sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)

    pred = model.predict(X_test)
    origanl_RMSE = np.sqrt(mean_squared_error(y_test, pred))
    origanl_r2 = r2_score(y_test, pred)


    pred_transform = scaler.inverse_transform(pred).flatten()
    te = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    #--------------------------------------------Test on Potatoes-------------------------------------------
    df2 = df.where(df['Area']=="Australia")
    df2 = df2.where(df['Crop']=="Potatoes")
    df2 = df2.dropna(how='any',axis=0) 
    df2['Year'] = pd.to_datetime(df2['Year'])
    df2.set_index("Year", inplace=True)
    data2 = df2["Crop_Yield"].astype(float).values.reshape(-1,1)
    data2 = scaler.fit_transform(data2)


    potatoeX, potatoey = pre_process(data2)
    potatoeX = potatoeX.reshape((potatoeX.shape[0], potatoeX.shape[1], 1))
    prediction_potatoe = model.predict(potatoeX)
    prediction_potatoe_actual = scaler.inverse_transform(prediction_potatoe)
    potatoey_actual = scaler.inverse_transform(potatoey.reshape(-1,1)).flatten()
    testScore_potatoe = np.sqrt(mean_squared_error(potatoey, prediction_potatoe))


    #--------------------------------------------Test on Wheat-------------------------------------------
    df3 = df.where(df['Area']=="Australia")
    df3 = df3.where(df['Crop']=="Wheat")
    df3 = df3.dropna(how='any',axis=0) 
    df3['Year'] = pd.to_datetime(df3['Year'])
    df3.set_index("Year", inplace=True)
    data3 = df3["Crop_Yield"].astype(float).values.reshape(-1,1)
    data3 = scaler.fit_transform(data3)

    wheatX, wheaty = pre_process(data3)
    wheatX = wheatX.reshape((wheatX.shape[0], wheatX.shape[1], 1))
    prediction_wheat = model.predict(wheatX)
    prediction_wheat_actual = scaler.inverse_transform(prediction_wheat)
    wheaty_actual = scaler.inverse_transform(wheaty.reshape(-1,1)).flatten()

    testScore_wheat = np.sqrt(mean_squared_error(wheaty, prediction_wheat))

    #--------------------------------------------Test on Potatoes (Bangladesh)-------------------------------------------
    df4 = df.where(df['Area']=="Bangladesh")
    df4 = df4.where(df['Crop']=="Potatoes")
    df4 = df4.dropna(how='any',axis=0) 
    df4['Year'] = pd.to_datetime(df4['Year'])
    df4.set_index("Year", inplace=True)
    data4 = df4["Crop_Yield"].astype(float).values.reshape(-1,1)
    data4 = scaler.fit_transform(data4)


    potatoe_bangladeshX, potatoey_bangladesh = pre_process(data4)
    potatoe_bangladeshX = potatoe_bangladeshX.reshape((potatoe_bangladeshX.shape[0], potatoe_bangladeshX.shape[1], 1))
    prediction_potatoe_bangladesh = model.predict(potatoe_bangladeshX)
    prediction_potatoe_bangladesh_actual = scaler.inverse_transform(prediction_potatoe_bangladesh)
    potatoey_bangladesh_actual = scaler.inverse_transform(potatoey_bangladesh.reshape(-1,1)).flatten()

    testScore_potatoe_bangladesh = np.sqrt(mean_squared_error(potatoey_bangladesh, prediction_potatoe_bangladesh))

    #--------------------------------------------Plotting Wheat (Bangladesh)-------------------------------------------
    df5 = df.where(df['Area']=="Bangladesh")
    df5 = df5.where(df['Crop']=="Wheat")
    df5 = df5.dropna(how='any',axis=0) 
    df5['Year'] = pd.to_datetime(df5['Year'])
    df5.set_index("Year", inplace=True)
    data5 = df5["Crop_Yield"].astype(float).values.reshape(-1,1)
    data5 = scaler.fit_transform(data5)

    wheat_bangladeshX, wheaty_bangladesh = pre_process(data5)
    wheat_bangladeshX = wheat_bangladeshX.reshape((wheat_bangladeshX.shape[0], wheat_bangladeshX.shape[1], 1))
    prediction_wheat_bangladesh = model.predict(wheat_bangladeshX)
    prediction_wheat_bangladesh_actual = scaler.inverse_transform(prediction_wheat_bangladesh)
    wheaty_bangladesh_actual = scaler.inverse_transform(wheaty_bangladesh.reshape(-1,1)).flatten()

    testScore_wheat_bangladesh = np.sqrt(mean_squared_error(wheaty_bangladesh, prediction_wheat_bangladesh))

    return jsonify({"x":dates_test.tolist(), "original_actual": te.tolist(), "original_pred": pred_transform.tolist(), "original_rmse": float(origanl_RMSE), "original_r2": float(origanl_r2),
                    "bang_pot_actual": potatoey_bangladesh_actual.tolist(), "bang_pot_pred": prediction_potatoe_bangladesh_actual.tolist(), "bang_pot_rmse": float(testScore_potatoe_bangladesh), "bang_pot_r2": float(r2_score(potatoey_bangladesh, prediction_potatoe_bangladesh)),
                    "bang_wheat_actual": wheaty_bangladesh_actual.tolist(), "bang_wheat_pred": prediction_wheat_bangladesh_actual.tolist(), "bang_wheat_rmse": float(testScore_wheat_bangladesh), "bang_wheat_r2": float(r2_score(wheaty_bangladesh, prediction_wheat_bangladesh))})

# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
