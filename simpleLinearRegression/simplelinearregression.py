import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)


#predicting the test set results
y_precit = regressor.predict(X_test);

#visualizing the traing set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of experience');
plt.ylabel('Salary')
plt.show()

#visualizing the test set results
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('years of experience');
plt.ylabel('Salary')
plt.show()