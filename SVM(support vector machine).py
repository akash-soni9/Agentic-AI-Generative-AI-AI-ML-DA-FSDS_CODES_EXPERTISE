import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\C\2nd - SVM\Social_Network_Ads.csv')

X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(probability=True, random_state=0)
classifier.fit(X_train, y_train)

#predict
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

cm = confusion_matrix(y_test, y_pred)
print("confusion metrix:", cm)

cr = classification_report(y_test, y_pred,)
print("report:", cr)


# validation on future data

dataset1 = pd.read_csv(r'C:\Users\akash\OneDrive\Desktop\FSDS\ML\C\2nd - SVM\Future prediction1.csv')
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, 2:].values

M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2['y_pred1'] = classifier.predict(M)

print(d2)


# AUC - ROC Curve
from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_prob)
auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'Support Vector Machine (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()






