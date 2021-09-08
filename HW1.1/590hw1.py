#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:02:07 2021

@author: ruitongliu
"""

# import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import json
from scipy.optimize import minimize

#read data
with open('/Users/ruitongliu/Desktop/weight.json') as fr:
    json_dict = json.load(fr)
    
xlabel = json_dict['xlabel']
ylabel = json_dict['ylabel']
is_adult = np.array(json_dict['is_adult'])
X = np.array(json_dict['x'])
y = np.array(json_dict['y'])

n = len(is_adult)
np.random.seed(42)
train_inds = np.random.choice(range(n), size = int(0.8*n), replace=False)
test_inds = np.array(list(set(range(n)) - set(train_inds)))

# break the data into 80% training 20% test
X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds]
is_adult_train, is_adult_test = is_adult[train_inds], is_adult[test_inds]

X_mean = X_train.mean()
X_std = X_train.std()
SS_X_train = (X_train - X_mean) / X_std
SS_X_test = (X_test - X_mean) / X_std

y_mean = y_train.mean()
y_std = y_train.std()
SS_y_train = (y_train - y_mean) / y_std
SS_y_test = (y_test - y_mean) / y_std

# developing a model
mask = X_train < 18
SS_X_train_fit = SS_X_train[mask]
SS_y_train_fit = SS_y_train[mask]


def OLS_pred(X_obs, m, b):
    return m * X_obs + b 

def cal_OLS_MSE(params, X_obs, y_obs):
    m, b = params
    y_pred = OLS_pred(X_obs, m, b)
    mse = np.mean((y_pred - y_obs)**2)
    return mse

result = minimize(cal_OLS_MSE, np.array([5,25]), args=(SS_X_train_fit, SS_y_train_fit))
m,b = result['x']

min_SS_X_train_fit = min(SS_X_train_fit)
max_SS_X_train_fit = max(SS_X_train_fit)

SS_X_temp = np.linspace(min_SS_X_train_fit, -1, 100)
SS_y_temp = OLS_pred(SS_X_temp, m, b )
X_temp = SS_X_temp * X_std + X_mean
y_temp = SS_y_temp * y_std + y_mean

# liner regression
plt.figure(figsize = (18,4))
plt.subplot(131)
plt.scatter(X,y,s = 5)
plt.plot(X_temp, y_temp, c = 'black', linewidth = 2)
plt.xlabel('age (year)')
plt.ylabel('weight (lb)')
plt.text(40, 120 , "y=f(x|p)=mx+b\n\np=({:.4},{:.4})".format(m,b), fontsize = 14)
plt.title('Linear', fontsize = 17)


def Log_pred(X_obs, A, w, x0, S):
    return A / (1 + np.exp(-(X_obs - x0)/w)) + S

def cal_Log_MSE(params, X_obs, y_obs):
    A, w, x0, S = params
    y_pred = Log_pred(X_obs, A, w, x0, S)
    mse = np.mean((y_pred - y_obs)**2)
    return mse


result = minimize(cal_Log_MSE, np.array([1,1,0,5]), args=(SS_X_train, SS_y_train))
A, w, x0, S = result['x']

min_SS_X_train = min(SS_X_train)
max_SS_X_train = max(SS_X_train)

SS_X_temp = np.linspace(min_SS_X_train, max_SS_X_train, 1000)
SS_y_temp = Log_pred(SS_X_temp, A, w, x0, S)
X_temp = SS_X_temp * X_std + X_mean
y_temp = SS_y_temp * y_std + y_mean

# logistic regression
plt.figure(figsize = (18,4))
plt.subplot(132)
plt.scatter(X,y,s = 5)
plt.plot(X_temp, y_temp, c = 'black', linewidth = 2)
plt.xlabel('age (year)')
plt.ylabel('weight (lb)')
plt.text(18, 100 , "y=f(x|p)=A/(1+exp(-(X-x0)/w))+S\n\np=({:.4},{:.4},{:.4},{:.4})".format(A, w, x0, S), 
         fontsize = 14)
plt.title('Logistic', fontsize = 17)


result = minimize(cal_Log_MSE, np.array([1,2,0,0.5]), args=(SS_y_train, is_adult_train))
A, w, x0, S = result['x']

min_SS_X_train = min(SS_y_train)
max_SS_X_train = max(SS_y_train)

SS_X_temp = np.linspace(min_SS_X_train, max_SS_X_train, 1000)
is_adult_temp = Log_pred(SS_X_temp, A, w, x0, S)

X_temp = SS_X_temp * y_std + y_mean

# predict logistic regression
plt.figure(figsize = (18,4))
plt.subplot(133)
plt.scatter(y, is_adult, s = 5)
plt.plot(X_temp, is_adult_temp, c = 'black', linewidth = 2)
plt.ylabel('ADULT=1 CHILD=0')
plt.xlabel('weight (lb)')
plt.text(30, 0.6 , "y=f(x|p)=A/(1+exp(-(X-x0)/w))+S\n\np=({:.3},{:.3},{:.3},{:.3})".format(A, w, x0, S), 
         fontsize = 14)
plt.title('Logistic', fontsize = 14)
plt.show()
