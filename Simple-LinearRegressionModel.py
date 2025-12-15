import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\Salary_Data.csv')


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Year of Experiece (Test Set)")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.show()


# Validation or Future data
m_slope = regressor.coef_
print(f"Coefficient: {m_slope}")

c_intercept = regressor.intercept_
print(f"Intercept: {c_intercept}")

y_12 = m_slope * 12 + c_intercept
print(y_12)

y_20 = m_slope * 20 + c_intercept
print(y_20)

# Training score = bias
bias_score = regressor.score(X_train, y_train)
print(bias_score)

variance_score = regressor.score(X_test, y_test)
print(variance_score)

# lets Implement stats to this model
dataset.mean()
dataset['Salary'].mean()
dataset['YearsExperience'].mean()

dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

from scipy.stats import variation
variation(dataset.values)
variation(dataset['Salary'])
variation(dataset['YearsExperience'])

dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])
dataset['Salary'].corr(dataset['Salary'])

dataset.skew()
dataset.sem()


import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])
stats.zscore(dataset['YearsExperience'])

#z_scores = dataset.apply(stats.zscore)
#print(z_scores)


# Anova - SST, SSR, SSE

y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean) ** 2)
print(SSR)

y = y[0:6]
SSE = np.sum((y - y_pred) ** 2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values - mean_total) ** 2)
print(SST)

r_square = 1 - (SSR / SST)

print(r_square)
print(bias_score)
print(variance_score)