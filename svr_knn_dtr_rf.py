import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\emp_sal.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

## SVR 
from sklearn.svm import SVR
regressor = SVR(degree=4, kernel='poly', gamma='auto', C=5.0)
regressor.fit(x, y)

y_pred_svr = regressor.predict([[6.5]])
print('SVR:',y_pred_svr)


# KNN regressor
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(weights='distance', algorithm='brute', n_neighbors=2)
knn_reg.fit(x, y)

y_pred_knn = knn_reg.predict([[6.5]])
print('KNN:',y_pred_knn)


#dtr- decision tree regresssion
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='poisson', splitter='best', max_depth=3, random_state=0)

dt_reg.fit(x, y)
dt_pred=dt_reg.predict([[6.5]])
print('DTR:',dt_pred)

# random forest regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=43, n_estimators=20)
rf_reg.fit(x, y)
rf_pred = rf_reg.predict([[6.5]])
print('RF:',rf_pred)














