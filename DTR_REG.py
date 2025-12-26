import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\emp_sal.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(criterion='poisson', random_state=0, max_depth=3, splitter='best')
dtr.fit(x, y)

plt.scatter(x, y, color='red')
plt.plot(x, dtr.predict(x), color='blue')
plt.title("Truth or Bluff (DTR)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

dtr_pred = dtr.predict([[6.5]])
print("DTR:",dtr_pred)