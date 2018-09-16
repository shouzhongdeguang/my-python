# Multiple linear regression
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset = pd.read_csv('../100-Days-Of-ML-Code/datasets/50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder#编码
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
#print(X)
#Avoiding Dummy Variable Trap 防止编码陷阱
X = X[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor1 = regressor.fit(X_train,Y_train)

Y_pred = regressor1.predict(X_test)
plt.scatter(Y_test,Y_pred)
#plt.show()
regressor = LinearRegression()
Y_pred = Y_pred.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
regressor = regressor.fit( Y_test,Y_pred)
plt.plot(Y_test,regressor.predict(Y_test))
plt.show()
from sklearn.metrics import r2_score
print(r2_score(Y_pred,Y_test))    
