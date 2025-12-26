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

# Visualization 
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

y_pred_svr = regressor.predict([[6.5]])
print('SVR:',y_pred_svr)