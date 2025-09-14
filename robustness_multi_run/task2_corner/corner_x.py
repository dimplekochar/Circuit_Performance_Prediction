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
X = data14nmco[:,[0,1,2,3,6]]
y = data14nmco[:,5]
tsplit = [0.7, 0.8, 0.9, 0.95]
with open('data/output_random_x.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['tsplit', 'algoseed', 'splitseed', 'mse', 'r2', 'mae', 'medae', 'mape'])
    for k in tsplit:
        for i in range(0, 50):
            for j in range(0, 50):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=j)
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.9, seed=i)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                MAE = mean_absolute_error(y_test, y_pred)
                MEDAE = median_absolute_error(y_test, y_pred)
                MAPE = mean_absolute_percentage_error(y_test, y_pred)
                writer.writerow([k, i, j, mse, r2, MAE, MEDAE, MAPE])
