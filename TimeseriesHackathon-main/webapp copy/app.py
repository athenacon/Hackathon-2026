from flask import Flask, jsonify, render_template, request
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import *
from sklearn.linear_model import LinearRegression
app = Flask(__name__)

# This route "/" is the default page, this will run when the browerser runs the URL
@app.route("/")
def index():
    return render_template("index.html")

# This route "/dropdown" is called by a script in the index.html to get the options for the dropdown menu
@app.route("/dropdown")
def dropdown():
    df = pd.read_csv("combined_data.csv") # Read the CSV file
    crops = df["Crop"].unique().tolist() # Get the unique values from the "Crop" column and convert to a list 

    options = [{"value": c, "label": c} for c in crops]# Create a list of dictionaries with "value" and "label" keys for each crop

    return jsonify(options)

# This is called by a acript in the index.html
@app.route("/runModelGlobal")
def runModelGlobal():
    def pre_process(X, y, look_back):
        X_out, y_out = [], []

        for i in range(look_back, len(X)):
            X_out.append(X[i - look_back:i, :])
            y_out.append(y[i, 0])

        return np.array(X_out), np.array(y_out)
    
    # Read the seed from the query string e.g. /run?seed=42, default to 42 if not provided
    seed = int(request.args.get("seed", 42))
    split_year = int(request.args.get("split_year", 2010))
    features_string = request.args.get("features", "")
    features = [f for f in features_string.split(",") if f]
    crop = request.args.get("status") # Read the status (crop) from the query string 

    # # dataset path
    dataset_path = "combined_data.csv"
    dataframe = pd.read_csv(dataset_path)

    df = dataframe.copy()
    df = df[df["Crop"] == crop] # Filter the DataFrame to include only rows where the "Crop" column matches the selected crop
    df["Area"] = df["Area"].astype("category")
    df["Crop"] = df["Crop"].astype("category")

    df["average_rain_fall_mm_per_year"] = pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")
    cols = [
    "Area",
    "Year",
    "Crop",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticide_amount",
    "Crop_Yield",
    ]
    df = df.dropna(subset=cols)
    df = df.sort_values('Year')

    train_df = df[df['Year'] < split_year]
    test_df = df[df['Year'] >= split_year]

    X_train = train_df[features]
    y_train = train_df["Crop_Yield"]

    X_test = test_df[features]
    y_test = test_df["Crop_Yield"]

    #Create, fit and run out model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed, enable_categorical=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # results /  actuals
    r2 = sklearn.metrics.r2_score(y_test, preds)
    mse = sklearn.metrics.mean_squared_error(y_test, preds)


    #--------------------------------------------------------------------------------------------------LSTM model----------------------------------------------------------------------------

    df1 = df.copy()
    for col in features:
        if df1[col].dtype == "object" or str(df1[col].dtype) == "category":
            le = LabelEncoder()
            df1[col] = le.fit_transform(df1[col])

    X_data = df1[features].values
    y_data = df1["Crop_Yield"].values.reshape(-1, 1)
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_data)
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y_data)

    look_back = look_back = max(1, df["Year"].max() - split_year) # Set look_back to the number of years in the test set, or at least 1
    split_index = int(len(X_scaled) * 0.7)

    X_train_raw = X_scaled[:split_index]
    X_test_raw  = X_scaled[split_index:]

    y_train_raw = y_scaled[:split_index]
    y_test_raw  = y_scaled[split_index:]

    X_lin_train, y_lin_train = pre_process(X_train_raw, y_train_raw, look_back)
    X_lin_test, y_lin_test   = pre_process(X_test_raw, y_test_raw, look_back)

    model_LSTM = Sequential()
    model_LSTM.add(LSTM(16, input_shape=(X_lin_train.shape[1], X_lin_train.shape[2])))
    model_LSTM.add(Dense(8))
    model_LSTM.add(Dense(1))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam')
    model_LSTM.fit(X_lin_train, y_lin_train, epochs=100, batch_size=10, verbose=2)

    pred = model_LSTM.predict(X_lin_test)
    pred_transform = target_scaler.inverse_transform(pred).flatten()
    te = target_scaler.inverse_transform(y_lin_test.reshape(-1,1)).flatten()
    LSTM_rmse = np.sqrt(sklearn.metrics.mean_squared_error(te, pred_transform))
    LSTM_r2 = sklearn.metrics.r2_score(te, pred_transform)

#------------------------------------------Linear Regression model----------------------------------------------------------------------------
    x_train_lin = X_train.apply(pd.to_numeric, errors='coerce')
    y_train_lin = y_train.apply(pd.to_numeric, errors='coerce')
    x_test_lin = X_test.apply(pd.to_numeric, errors='coerce')
    y_test_lin = y_test.apply(pd.to_numeric, errors='coerce')
    x_train_lin.fillna(0, inplace=True)
    y_train_lin.fillna(0, inplace=True)
    x_test_lin.fillna(0, inplace=True)
    y_test_lin.fillna(0, inplace=True)
    
    lin_model = LinearRegression()
    lin_model.fit(x_train_lin, y_train_lin)
    lin_preds = lin_model.predict(x_test_lin)

    lin_r2 = sklearn.metrics.r2_score(y_test_lin, lin_preds)
    lin_mse = sklearn.metrics.mean_squared_error(y_test_lin, lin_preds)
#---------------------------------------------------------Resulsts---------------------------------------------------------------------------------------------------------------------------------------
    results = [
        {"index": i, 
         "actual": round(float(y_test.iloc[i]), 2), 
         "predicted": round(float(preds[i]), 2)
        }
        for i in range(min(50, len(y_test)))
    ]

    results_LSTM = [
        {"index": i, 
         "actual": round(float(te[i]), 2), 
         "predicted": round(float(pred_transform[i]), 2)
        }
        for i in range(min(50, len(te)))
    ]

    results_linear = [
        {"index": i, 
         "actual": round(float(y_test_lin.iloc[i]), 2), 
         "predicted": round(float(lin_preds[i]), 2)
        }
        for i in range(min(50, len(y_test_lin)))
    ]

    # metrics
    metrics = {
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "LSTM_rmse": round(LSTM_rmse, 4),
        "LSTM_r2": round(LSTM_r2, 4),
        "lin_r2": round(lin_r2, 4),
        "lin_mse": round(lin_mse, 4)
    }

    #feature importance results
    importances = [
        {"feature": col, "importance": round(float(v), 4)}
        for col, v in zip(X_train.columns, model.feature_importances_)
    ]
    #return these results to script in index.html
    return jsonify({"results": results, "importances": importances, "metrics": metrics, "results_LSTM": results_LSTM, "results_linear": results_linear})


# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
