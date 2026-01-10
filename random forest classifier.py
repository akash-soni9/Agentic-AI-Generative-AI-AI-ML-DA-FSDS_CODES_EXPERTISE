import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\akash\OneDrive\Desktop\FSDS\ML\C\3rd - KNN\Social_Network_Ads.csv")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
'''
# Feature scaling - does not required scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)

# ROC-AUC curve
from sklearn.metrics import roc_auc_score, roc_curve
# Predict probabilities
y_pred_prob = classifier.predict_proba(X_test)[:, 1]
# AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', linewidth=2,
         label=f'Random Forest (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
