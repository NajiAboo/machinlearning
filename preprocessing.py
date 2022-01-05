import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/Data.csv")
print(dataset)

#Split to X and y

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

print("*"*50)
print("X")
print(X)

print("*"*50)
print("y")
print(y)

# Missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])
print("Transformed values")
print(X[:,1:])

#Encoding Categorical variable 
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder


ct = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder="passthrough")
#columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
print(X)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

#split the data
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train)


#feature scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X[:,3:] = sc.fit_transform(X[:,3:])
print(X)





