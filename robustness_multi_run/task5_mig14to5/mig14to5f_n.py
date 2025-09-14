import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow
import csv 
import gc
import time
import os
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
data_14nm = data14nmsc
data_5nm = data5nmsc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
X_14nm = data14nmsc[:,0:4]
X_5nm = data5nmsc[:,0:4]
y_14nm = data14nmsc[:,4]
for i in range(data5nmsc.shape[0]):
    data5nmsc[i, 4] /= 1e9
y_5nm = data5nmsc[:,4]
file_path = "data/dataf.csv"
tsplit = [0.7, 0.8, 0.9, 0.95]
with open('data/output_randomf_n.csv', mode='a', newline='', buffering=1) as file:
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
        X_train_14nm, X_test_14nm, y_train_14nm, y_test_14nm = train_test_split(X_14nm, y_14nm, test_size=0.2, random_state=j)
        X_train_5nm, X_test_5nm, y_train_5nm, y_test_5nm = train_test_split(X_5nm, y_5nm, test_size=k, random_state=j)
        scaler = StandardScaler()
        X_train_scaled_14nm = scaler.fit_transform(X_train_14nm)
        X_test_scaled_14nm = scaler.transform(X_test_14nm)
        X_train_scaled_5nm = scaler.transform(X_train_5nm)
        X_test_scaled_5nm = scaler.transform(X_test_5nm)
        model_14nm = Sequential()
        model_14nm.add(Dense(64, input_dim=X_train_scaled_14nm.shape[1], activation='relu'))
        model_14nm.add(Dense(64, activation='relu'))
        model_14nm.add(Dense(64, activation='relu'))
        model_14nm.add(Dense(1, activation='linear'))
        model_14nm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model_14nm.fit(X_train_scaled_14nm, y_train_14nm, epochs=500, batch_size=32, validation_data=(X_test_scaled_14nm, y_test_14nm), verbose=0)
        transfer_model = model_14nm
        for layer in transfer_model.layers[:-2]:
            layer.trainable = False
        transfer_model.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')
        transfer_model.fit(X_train_scaled_5nm, y_train_5nm, epochs=400, batch_size=16, verbose=0)
        y_pred_5nm = transfer_model.predict(X_test_scaled_5nm)
        mse = mean_squared_error(y_test_5nm, y_pred_5nm)
        r2 = r2_score(y_test_5nm, y_pred_5nm)
        MAE = mean_absolute_error(y_test_5nm, y_pred_5nm)
        MEDAE = median_absolute_error(y_test_5nm, y_pred_5nm)
        MAPE = mean_absolute_percentage_error(y_test_5nm, y_pred_5nm)
        writer.writerow([k, i, j, mse, r2, MAE, MEDAE, MAPE])
        del model_14nm
        del transfer_model
        tensorflow.keras.backend.clear_session()
        gc.collect() 
        df = df.iloc[1:]
        df.to_csv(file_path, index=False)
        time.sleep(0.5) 
