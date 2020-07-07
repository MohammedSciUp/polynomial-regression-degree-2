#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_excel('D:\\1\\co2.xlsx')
print(df)
p_df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION','CO2EMISSIONS']]
x = df['ENGINESIZE']
y = df['CO2EMISSIONS']
print('======================================================================')
print(p_df.head(10))

#mission values with respect to Engine size scatter plot 
plt.figure(figsize=(5,5))
plt.plot(x,y,marker='o',markersize=2,linestyle=' ') #just another way of scatter plt , 
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

print('=======================================================================')

#data spliting 

split = np.random.rand(len(p_df))>0.8
train_data = p_df[split]
test_data  = p_df[~split]

print('train_data: ', train_data)
print('=======================================================================')
print('test_data: ', test_data)
print('***===================================================================')

train_x = np.asanyarray(train_data[['ENGINESIZE']])
train_y = np.asanyarray(train_data[['CO2EMISSIONS']])
print('train_x: ',train_x)
print('=======================================================================')
print('train_y: ',train_y)
print('*****===================================================================')

print('=======================================================================')

test_x = np.asanyarray(test_data[['ENGINESIZE']])
test_y = np.asanyarray(test_data[['CO2EMISSIONS']])
print('test_x: ',test_x)
print('=======================================================================')
print('test_y: ',test_y)
print('*****===================================================================')

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)
print('=======================================================')
linear_reg_prob = linear_model.LinearRegression()
train_y_ = linear_reg_prob.fit(train_x_poly, train_y)
# The thetas  and y-intercept 
print ('thetas: ', linear_reg_prob.coef_,'y_Intercept: ',linear_reg_prob.intercept_)
print ('=======================================================')

# plotting fitting results 
plt.scatter(train_data.ENGINESIZE, train_data.CO2EMISSIONS,  color='#c20641')
X_ = np.arange(0, 20, 0.1)
y_ = linear_reg_prob.intercept_[0]+ linear_reg_prob.coef_[0][1]*X_+ linear_reg_prob.coef_[0][2]*np.power(X_, 2)
plt.plot(X_, y_, '#e635e6' )
plt.xlabel("Enginesize")
plt.ylabel("Emission")
plt.grid()
plt.show()
print ('=======================================================')
# (5) observe the error


test_poly = poly.fit_transform(test_x)
test_y_ = linear_reg_prob.predict(test_poly)

print("MSR: %.2f" % np.mean(np.absolute(test_y_ - test_y)),"MSE: %.2f" % np.mean((test_y_ - test_y) ** 2),"R2-score: %.2f" % r2_score(test_y_ , test_y) )

