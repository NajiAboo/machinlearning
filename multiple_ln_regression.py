import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('data/50_Startups.csv')
print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X =   np.array(ct.fit_transform(dataset))

print(X)

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 )


regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print(y_pred)

comp = np.concatenate( (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(comp)
