import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\emp_sal.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title("Linear regression")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

# we will be using polynomial model 
# linear model not enouggh to polynomial data

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title("join or not join (polynomial Regression)")
plt.xlabel("postion")
plt.ylabel("salary")
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)











