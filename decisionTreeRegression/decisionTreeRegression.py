
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#fitting the decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


#predecting a new result
y_pred =regressor.predict(np.array([[6.5]]))


#visualizing the SVR results
#higher resolution and smoother curver
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title('truth or bluff DTR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


