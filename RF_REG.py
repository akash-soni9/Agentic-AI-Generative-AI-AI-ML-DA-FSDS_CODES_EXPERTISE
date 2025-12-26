import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\emp_sal.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=43, n_estimators=20)
rf_reg.fit(x, y)

# Visulization
plt.scatter(x, y, color='red')
plt.plot(x, rf_reg.predict(x), color = 'blue')
plt.title("Truth or Bluff (RF)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)