import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\emp_sal.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# KNN
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5, algorithm='brute', weights='distance')
knn_reg.fit(x,y)

#Visualization
plt.scatter(x, y, color='red')
plt.plot(x, knn_reg.predict(x), color='blue')
plt.title("Truth or Bluff (KNN)")
plt.xlabel("Position")
plt.ylabel("salary")
plt.show()

knn_pred = knn_reg.predict([[6.5]])
print("KNN:",knn_pred)