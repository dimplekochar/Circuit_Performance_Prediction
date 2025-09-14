import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error 
import csv
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
X_14nm1 = data14nmsc[:,0:4]
X_5nm = data5nmsc[:,0:4]
y_14nm1 = data14nmsc[:,5]
for i in range(data5nmsc.shape[0]):
    data5nmsc[i, 4] /= 1e9
y_5nm = data5nmsc[:,5]
tsplit = [0.7, 0.8, 0.9, 0.95]
with open('data/output_random_x.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['tsplit', 'algoseed', 'splitseed', 'mse', 'r2', 'mae', 'medae', 'mape'])
    for k in tsplit:
        for i in range(0, 50):
            for j in range(0, 50):
                X_train_5nm, X_test_5nm, y_train_5nm, y_test_5nm = train_test_split(X_5nm, y_5nm, test_size=k, random_state=j)
                scaler = StandardScaler()
                X_14nm = X_14nm1
                X_14nm = scaler.fit_transform(X_14nm)
                X_train_5nm = scaler.transform(X_train_5nm)
                X_test_5nm = scaler.transform(X_test_5nm)
                scaler1 = StandardScaler()
                y_14nm = y_14nm1
                y_14nm = scaler1.fit_transform(y_14nm.reshape(-1, 1)).flatten()
                y_train_5nm = scaler1.transform(y_train_5nm.reshape(-1, 1)).flatten()
                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.9, random_state=i)
                model.fit(X_14nm, y_14nm)
                model.fit(X_train_5nm, y_train_5nm)
                y_pred_scaled2 = model.predict(X_test_5nm)
                y_pred2 = scaler1.inverse_transform(y_pred_scaled2.reshape(-1, 1))
                mse = mean_squared_error(y_test_5nm, y_pred2)
                r2 = r2_score(y_test_5nm, y_pred2)
                MAE = mean_absolute_error(y_test_5nm, y_pred2)
                MEDAE = median_absolute_error(y_test_5nm, y_pred2)
                MAPE = mean_absolute_percentage_error(y_test_5nm, y_pred2)
                writer.writerow([k, i, j, mse, r2, MAE, MEDAE, MAPE])
