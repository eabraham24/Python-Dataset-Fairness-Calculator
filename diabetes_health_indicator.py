import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import os


from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif

names = ['Diabetes_binary','HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits', 'Veggies', 'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth', 'MentHlth','PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

subPath = 'archive\diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
fullPath = os.path.dirname(__file__)+'\\'+subPath

print('\n********************************************************************\n')
print('Full Path ***************', fullPath)
print('Version********',pd.__version__,'\n')
print('\n********************************************************************\n')




data = pd.read_csv(fullPath)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[['Diabetes_binary','HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth', 'PhysHlth','DiffWalk','Sex','Age','Education','Income']], data.Diabetes_binary, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

from sklearn import metrics
predict_test = model.predict(X_test)
print(metrics.accuracy_score(y_test, predict_test))

