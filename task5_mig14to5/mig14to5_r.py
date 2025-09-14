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
from sklearn.ensemble import RandomForestRegressor
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
X_14nm = data14nmsc[:,0:4]
X_5nm = data5nmsc[:,0:4]
y_14nm = data14nmsc[:,5]
for i in range(data5nmsc.shape[0]):
    data5nmsc[i, 4] /= 1e9
y_5nm = data5nmsc[:,5]
X_train_5nm, X_test_5nm, y_train_5nm, y_test_5nm = train_test_split(X_5nm, y_5nm, test_size=0.7, random_state=42)
scaler = StandardScaler()
X_14nm = scaler.fit_transform(X_14nm)
X_train_5nm = scaler.transform(X_train_5nm)
X_test_5nm = scaler.transform(X_test_5nm)
scaler1 = StandardScaler()
y_14nm = scaler1.fit_transform(y_14nm.reshape(-1, 1)).flatten()
y_train_5nm = scaler1.transform(y_train_5nm.reshape(-1, 1)).flatten()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_14nm, y_14nm)
model.fit(X_train_5nm, y_train_5nm)
y_pred_scaled2 = model.predict(X_test_5nm)
y_pred2 = scaler1.inverse_transform(y_pred_scaled2.reshape(-1, 1))
mse = mean_squared_error(y_test_5nm, y_pred2)
r2 = r2_score(y_test_5nm, y_pred2)
MAE = mean_absolute_error(y_test_5nm, y_pred2)
MEDAE = median_absolute_error(y_test_5nm, y_pred2)
MAPE = mean_absolute_percentage_error(y_test_5nm, y_pred2)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
print(f"MAE: {MAE}")
print(f"MEDAE: {MEDAE}")
print(f"MAPE: {MAPE}")
print(mse)
print(r2)
print(MAE)
print(MEDAE)
print(MAPE)
relative_bias = np.mean((y_pred2.squeeze() - y_test_5nm.squeeze()) / y_test_5nm.squeeze()) * 100
relative_bias_mape = np.mean(np.abs((y_pred2.squeeze() - y_test_5nm.squeeze()) / y_test_5nm.squeeze())) * 100
print("Relative Bias (%):", relative_bias)
print("Relative Bias MAPE (%):", relative_bias_mape)