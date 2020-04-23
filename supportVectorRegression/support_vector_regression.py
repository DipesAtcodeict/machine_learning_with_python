# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:19:30 2020

@author: dipes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))


#fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#predecting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))


#visualizing the SVR results
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title('truth or bluff SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
