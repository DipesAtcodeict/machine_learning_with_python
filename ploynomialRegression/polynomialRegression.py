
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression();
lin_reg.fit(X,y);


#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#visualizing linear regression results
plt.scatter(X,y, color= 'red')
plt.plot(X, lin_reg.predict(X),color="blue")
plt.title('Truth or Bluff (Linear Regression) ')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualizing polynomial regression results
plt.scatter(X,y, color= 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color="blue")
plt.title('Truth or Bluff (Ploynomial Regression) ')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#predicting a new result with Linear Regression
sal_lin = lin_reg.predict([[6.5]])

#predicting with polynomial regression
sal_pol = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


