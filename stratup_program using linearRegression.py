import pandas as pd
import numpy as np
data=pd.read_csv('50_Startups_lyst7770.csv')
#print(data)
#x=data[['YearsExperience']].values
#slicing
y=data.iloc[:,-1].values
x=data.iloc[:,:-1].values
#print(x)
#print(y)

#encoding using oneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A=make_column_transformer((OneHotEncoder(categories="auto"),[3]),remainder="passthrough")
x=A.fit_transform(x)


#splitting the data training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
rt=LinearRegression()
rt.fit(x_train,y_train)
y_pred=rt.predict(x_test)
print(y_pred)

print(rt.coef_)
print(rt.intercept_)




