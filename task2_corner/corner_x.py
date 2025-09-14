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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.9)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MEDAE = median_absolute_error(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
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
relative_bias = np.mean((y_pred.squeeze() - y_test.squeeze()) / y_test.squeeze()) * 100
relative_bias_mape = np.mean(np.abs((y_pred.squeeze() - y_test.squeeze()) / y_test.squeeze())) * 100
print("Relative Bias (%):", relative_bias)
print("Relative Bias MAPE (%):", relative_bias_mape)
