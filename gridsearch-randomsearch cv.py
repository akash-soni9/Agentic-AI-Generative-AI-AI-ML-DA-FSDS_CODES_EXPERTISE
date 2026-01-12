import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\C\logit classification.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(C = 1.0, kernel='rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("classification report:",cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

cr = classification_report(y_test, y_pred)
print("classification report:", cr)

# Training score = bias
bias_score = classifier.score(X_train, y_train)
print(bias_score)

#variance
variance_score = classifier.score(X_test, y_test)
print(variance_score)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv = 5)
print(f"accuracies : {accuracies.mean()*100:.4f}")
print(f" std accuracies : {accuracies.std()*100:.4f}")



# grid search cv
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C':[1, 10, 100, 1000], 'kernel': ['linear']},
    {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    ]

grid_search = GridSearchCV(
    estimator = classifier,
    param_grid = parameters,
    scoring='accuracy', cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print(f"best accuracy : {best_accuracy*100:.4f}")
print(f"best params : {best_params}")


#random search cv
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

parameters_random = {
    'C':uniform(1, 1000), 
    'kernel': ['linear', 'rbf'], 
    'gamma': uniform(0.01, 1)
    }
    
    

random_search = RandomizedSearchCV(
    estimator = classifier,
    param_distributions= parameters_random,
    n_iter=  50,
    scoring='accuracy',
    cv = 10,
    random_state=0,
    n_jobs=-1
 )

random_search = random_search.fit(X_train, y_train)
best_accuracy = random_search.best_score_
best_params = random_search.best_params_

print(f"best accuracy random search : {best_accuracy*100:.4f}%")
print(f"best params random search: {best_params}")


