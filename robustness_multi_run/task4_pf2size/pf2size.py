import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from xgboost import XGBRegressor, XGBClassifier
import tensorflow
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error 
import gc
import csv
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def round_to_nearest(arr, rounding_vals):
    diffs = np.abs(arr[:, None] - rounding_vals)
    idx_min = diffs.argmin(axis=1)
    return rounding_vals[idx_min]
ring14nmsc = "data/14nm_schematic.xlsx"
ring14nmla = "data/14nmdata_ro.xlsx"
ring14nmme = "data/14nm_chip.xlsx"
ring14nmco = "data/ro14nm_corners_schematic_1.xlsx"
ring5nmsc = "data/ro5nm_schematic.xlsx"
data14nmsc = pd.read_excel(ring14nmsc).iloc[:, :6]
data14nmsc = data14nmsc.apply(pd.to_numeric, errors='coerce')
data14nmsc = data14nmsc.dropna()
data14nmsc = data14nmsc.to_numpy(dtype='f')
data14nmla = pd.read_excel(ring14nmla).iloc[:, :6]
data14nmla = data14nmla.apply(pd.to_numeric, errors='coerce')
data14nmla = data14nmla.dropna()
data14nmla = data14nmla.to_numpy(dtype='f')
data14nmme = np.array((pd.read_excel(ring14nmme).to_numpy())[:,:6], dtype = 'f')
data14nmco = pd.read_excel(ring14nmco).iloc[:, :7]
data14nmco = data14nmco.apply(pd.to_numeric, errors='coerce')
data14nmco = data14nmco.dropna()
data14nmco = data14nmco.to_numpy(dtype='f')
data5nmsc = pd.read_excel(ring5nmsc).iloc[:, :6]
data5nmsc = data5nmsc.apply(pd.to_numeric, errors='coerce')
data5nmsc = data5nmsc.dropna()
data5nmsc = data5nmsc.to_numpy(dtype='f')
x1 = data14nmsc[:,4]
x2 = data14nmsc[:,5]
y1 = data14nmsc[:,1]
y2 = data14nmsc[:,2]
y3 = data14nmsc[:,0]
y4 = data14nmsc[:,3]
X = np.column_stack((x1, x2))
y = np.column_stack((y1, y2, y3, y4))
file_path = "data/data.csv"
tsplit = [0.7, 0.8, 0.9, 0.95]
with open('data/output_random_n.csv', mode='a', newline='', buffering=1) as file:
    writer = csv.writer(file)
    while True:
        df = pd.read_csv(file_path)
        if df.empty:
            print("No more data to process.")
            break
        k, i, j = df.iloc[0]
        i = int(i)
        j = int(j)
        tensorflow.keras.utils.set_random_seed(i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=j)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(y_train.shape[1]))
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
        model.fit(X_train_scaled, y_train_scaled, epochs=800, batch_size=32, validation_split=0.2, verbose=0)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        ypred1 = np.array([-40, -20, 0, 20, 40, 60, 80, 100])  
        ypred2 = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) 
        ypred3 = np.array([s1, s2, s3, s4, s5, s6, s7, s8]) #list your RO sizes here
        ypred4 = np.array([st1, st2, st3]) #list your RO number of stages here
        y_pred_round = y_pred
        y_pred_round[:,0] = round_to_nearest(y_pred_round[:,0], ypred1)
        y_pred_round[:,1] = round_to_nearest(y_pred_round[:,1], ypred2)
        y_pred_round[:,2] = round_to_nearest(y_pred_round[:,2], ypred3)
        y_pred_round[:,3] = round_to_nearest(y_pred_round[:,3], ypred4)
        mse1 = mean_squared_error(y_test[:,0], y_pred_round[:,0])
        r21 = r2_score(y_test[:,0], y_pred_round[:,0])
        MAE1 = mean_absolute_error(y_test[:,0], y_pred_round[:,0])
        MEDAE1 = median_absolute_error(y_test[:,0], y_pred_round[:,0])
        MAPE1 = mean_absolute_percentage_error(y_test[:,0], y_pred_round[:,0])
        mse2 = mean_squared_error(y_test[:,1], y_pred_round[:,1])
        r22 = r2_score(y_test[:,1], y_pred_round[:,1])
        MAE2 = mean_absolute_error(y_test[:,1], y_pred_round[:,1])
        MEDAE2 = median_absolute_error(y_test[:,1], y_pred_round[:,1])
        MAPE2 = mean_absolute_percentage_error(y_test[:,1], y_pred_round[:,1])
        mse3 = mean_squared_error(y_test[:,2], y_pred_round[:,2])
        r23 = r2_score(y_test[:,2], y_pred_round[:,2])
        MAE3 = mean_absolute_error(y_test[:,2], y_pred_round[:,2])
        MEDAE3 = median_absolute_error(y_test[:,2], y_pred_round[:,2])
        MAPE3 = mean_absolute_percentage_error(y_test[:,2], y_pred_round[:,2])
        mse4 = mean_squared_error(y_test[:,3], y_pred_round[:,3])
        r24 = r2_score(y_test[:,3], y_pred_round[:,3])
        MAE4 = mean_absolute_error(y_test[:,3], y_pred_round[:,3])
        MEDAE4 = median_absolute_error(y_test[:,3], y_pred_round[:,3])
        MAPE4 = mean_absolute_percentage_error(y_test[:,3], y_pred_round[:,3])
        writer.writerow([k, i, j, mse1, r21, MAE1, MEDAE1, MAPE1, mse2, r22, MAE2, MEDAE2, MAPE2, mse3, r23, MAE3, MEDAE3, MAPE3, mse4, r24, MAE4, MEDAE4, MAPE4])
        del model
        tensorflow.keras.backend.clear_session()
        gc.collect() 
        df = df.iloc[1:]
        df.to_csv(file_path, index=False)
        time.sleep(0.5)  
