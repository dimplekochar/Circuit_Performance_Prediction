import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error 
import matplotlib.pyplot as plt
from  xgboost  import  XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow
import csv
import gc
import time
ring14nmsc = "data/14nm_schematic.xlsx"
ring14nmla = "data/14nmdata_ro.xlsx"
ring14nmme = "data/14nm_chip.xlsx"
ring14nmco = "data/ro14nm_corners_schematic_1.xlsx"
ring5nmsc = "data/ro5nm_schematic.xlsx"
data14nmsc = pd.read_excel(ring14nmsc).iloc[:, :6]
data14nmsc = data14nmsc.apply(pd.to_numeric, errors='coerce')
data14nmsc = data14nmsc.dropna()
data14nmsc = data14nmsc.to_numpy(dtype='f')
data14nmla = pd.read_excel(ring14nmla).iloc[:, [0, 1, 2, 3, 5]]
data14nmla = data14nmla.apply(pd.to_numeric, errors='coerce')
data14nmla = data14nmla.dropna()
data14nmla = data14nmla.to_numpy(dtype='f')
data14nmme = pd.read_excel(ring14nmme, 'Chip2').iloc[:, [0, 1, 2, 3, 5]]
data14nmme = data14nmme.apply(pd.to_numeric, errors='coerce')
data14nmme = data14nmme.dropna()
data14nmme = data14nmme.to_numpy(dtype='f')
data14nmco = pd.read_excel(ring14nmco).iloc[:, :7]
data14nmco = data14nmco.apply(pd.to_numeric, errors='coerce')
data14nmco = data14nmco.dropna()
data14nmco = data14nmco.to_numpy(dtype='f')
data5nmsc = pd.read_excel(ring5nmsc).iloc[:, :6]
data5nmsc = data5nmsc.apply(pd.to_numeric, errors='coerce')
data5nmsc = data5nmsc.dropna()
data5nmsc = data5nmsc.to_numpy(dtype='f')
X = data14nmsc[:,[0,1,2,3,5]]
y = data14nmla[:,4]
tsplit = [0.7, 0.8, 0.9, 0.95]
k=0.7
i=42
j=42
tensorflow.keras.utils.set_random_seed(i)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=j)
scaler = StandardScaler()
scaler1 = StandardScaler()
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train_scaled = np.zeros_like(X_train)
X_test_scaled = np.zeros_like(X_test)
X_train_scaled[:,0:4] = scaler.fit_transform(X_train[:,0:4])
X_train_scaled[:,4] = scaler1.fit_transform(X_train[:, 4].reshape(-1, 1)).flatten()
X_scaled = np.zeros_like(X)
X_scaled[:,0:4] = scaler.fit_transform(X[:,0:4])
X_scaled[:,4] = scaler1.fit_transform(X[:, 4].reshape(-1, 1)).flatten()
X_test_scaled[:,0:4] = scaler.transform(X_test[:,0:4])
X_test_scaled[:,4] = scaler1.transform(X_test[:, 4].reshape(-1, 1)).flatten()
y_train_scaled = scaler1.fit_transform(y_train.reshape(-1, 1)).flatten()
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
model.fit(X_train_scaled, y_train_scaled, epochs=300, batch_size=32, verbose=0)
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler1.inverse_transform(y_pred_scaled)
X = data14nmsc[:,[0,1,2,3]]
y = y_pred
Xme = data14nmme[:,0:4]
yme = data14nmme[:,4]*1e-3
tsplit = [0.7, 0.8, 0.9, 0.95]
with open('data/output_random_n.csv', mode='a', newline='', buffering=1) as file:
    writer = csv.writer(file)
    writer.writerow(['tsplit', 'algoseed', 'splitseed', 'mse', 'r2', 'mae', 'medae', 'mape'])
    for k in tsplit:
        for i in range(42, 43):
            for j in range(42, 43):
                tensorflow.keras.utils.set_random_seed(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=j)
                X_train_me, X_test_me, y_train_me, y_test_me = train_test_split(Xme, yme, test_size=k, random_state=j)
                scaler = StandardScaler()
                scaler1 = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_scaled_me = scaler.transform(X_train_me)
                X_test_scaled_me = scaler.transform(X_test_me)
                y_train_scaled = scaler1.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test_scaled = scaler1.transform(y_test.reshape(-1, 1)).flatten()
                y_train_scaled_me = scaler1.transform(y_train_me.reshape(-1, 1)).flatten()
                model = Sequential()
                model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(1, activation='linear'))
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                model.fit(X_train_scaled, y_train_scaled, epochs=500, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), verbose=0)
                transfer_model = model
                for layer in transfer_model.layers[:-2]:
                    layer.trainable = False
                transfer_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
                transfer_model.fit(X_train_scaled_me, y_train_scaled_me, epochs=800, batch_size=16, verbose=0)
                y_pred_scaled_5nm = transfer_model.predict(X_test_scaled_me)
                y_pred_5nm = scaler1.inverse_transform(y_pred_scaled_5nm)
                y_pred_scaled = model.predict(X_test_scaled_me)
                y_pred = scaler1.inverse_transform(y_pred_scaled)
                mse = mean_squared_error(y_test_me, y_pred)
                r2 = r2_score(y_test_me, y_pred)
                MAE = mean_absolute_error(y_test_me, y_pred)
                MEDAE = median_absolute_error(y_test_me, y_pred)
                MAPE = mean_absolute_percentage_error(y_test_me, y_pred)
                writer.writerow([k, i, j, mse, r2, MAE, MEDAE, MAPE])
                del model
                tensorflow.keras.backend.clear_session()
                gc.collect()  
