# linear regression
#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]
#splitting datasets into train and test datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

#fitting linear regression model in training sets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set
y_pred = regressor.predict(x_test)
 
#visualisation of training sets 
plt.scatter(x_train,y_train, color ='red')
plt.plot(x_train,regressor.predict(x_train),color = 'black')
plt.title('salary vs experience(training set)' ,color = 'yellow')

plt.xlabel('experience' ,color = 'brown')
plt.ylabel('salary',color = 'green')

# visualisation of test sets
plt.scatter(x_test,y_test,color = 'blue')
plt.plot(x_train,regressor.predict(x_train) ,color = 'blue') #we will not change this by test because our regressor already leaarned by xtrai and ytrain so no need to change
plt.title('salary vs experience(test set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()