import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data/Salary_Data.csv')
print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)



X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
print(X_train)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor)

y_pred = regressor.predict(X_test)
print((y_pred))

print(X_test)

#Train
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')
plt.show()

#Test
plt.scatter(X_test,y_test,color='red')
# no need to change below line as the equation is same
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')
plt.show()
