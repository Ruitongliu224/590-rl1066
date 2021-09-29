#!/usr/bin/env python
# coding: utf-8


#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#--------------------------------

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json

#------------------------
#CODE PARAMETERS
#------------------------
#USER PARAMETERS
IPLOT=True

PARADIGM='batch'

model_type=["linear"]

I_NORMALZIE = True

#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]

#------------------------
#GET DATA
#------------------------
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv(url, 
                 names = column_names, 
                 na_values = "?",
                 comment = '\t',
                 sep=" ",
                 skipinitialspace=True)

import Seaborn_visualizer as SBV
SBV.get_pd_info(df)
SBV.pd_general_plots(df, HUE='Origin')

x_col = [1,2,3,4,5];
y_col = [0]; xy_col = x_col + y_col
X_KEYS = SBV.index_to_keys(df, x_col)
Y_KEYS = SBV.index_to_keys(df, y_col)

#CONVERT SELECT DF TO NP
x = df[X_KEYS].to_numpy()
y = df[Y_KEYS].to_numpy()

#REMOVE NAN IF PRESENT
xtmp = []; ytmp=[];
for i in range(0, len(x)):
    if(not "nan" in str(x[i])):
        xtmp.append(x[i])
        ytmp.append(y[i])

#MAKE ROWS-SAMPLE DIMENSION (TRANSPOSE)
X = np.array(xtmp); Y=np.array(ytmp)
NFIT = X.shape[1]+1

# SIGMOID
def S(x): return 1.0/(1.0+np.exp(-x))

print("-----------INPUT INFO-------------")
print("X shape:", X.shape); print("Y shape:", Y.shape, '\n')

#TAKE MEAN AND STD DOWN COLUMNS
XMEAN = np.mean(X, axis=0); XSTD = np.std(X, axis=0)
YMEAN = np.mean(Y, axis=0); YSTD = np.std(Y, axis=0)

#NORMALIZE 
if(I_NORMALZIE):
    X=(X-XMEAN)/XSTD;  Y=(Y-YMEAN)/YSTD 
    I_UNNORMALZIE = True
else: 
    I_UNNORMALZIE = False
    

#------------------------
#PARTITION DATA
#------------------------

f_train=0.8; f_val=0.15; f_test=0.05;

if(f_train+f_val+f_test != 1.0):
	raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

#PARTITION DATA
rand_indices = np.random.permutation(X.shape[0])
CUT1=int(f_train*X.shape[0]); 
CUT2=int((f_train+f_val)*X.shape[0]); 
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print("test_idx shape:" ,test_idx.shape)

#------------------------
#MODEL
#------------------------
def model(x,p):
    linear = p[0] + np.matmul(x,p[1:].reshape(NFIT-1,1))
    if(model_type=="linear"):   return linear
    if(model_type=="logistic"): return  S(linear)

#FUNCTION TO MAKE VARIOUS PREDICTIONS FOR GIVEN PARAMETERIZATION
def predict(p):
    global YPRED_T,YPRED_V,YPRED_TEST,MSE_T,MSE_V
    YPRED_T=model(X[train_idx],p)
    YPRED_V=model(X[val_idx],p)
    YPRED_TEST=model(X[test_idx],p)
    MSE_T=np.mean((YPRED_T-Y[train_idx])**2.0)
    MSE_V=np.mean((YPRED_V-Y[val_idx])**2.0)

#------------------------
#LOSS FUNCTION
#------------------------
def loss(p,index_2_use):
    errors=model(X[index_2_use],p)-Y[index_2_use]  #VECTOR OF ERRORS
    training_loss=np.mean(errors**2.0)                #MSE
    return training_loss

#------------------------
#MINIMIZER FUNCTION
#------------------------
def minimizer(f,xi, algo='GD', LR=5):
    global epoch,epochs, loss_train,loss_val 
    # x0=initial guess, (required to set NDIM)
    # algo=GD or MOM
    # LR=learning rate for gradient decent

    #PARAM
    iteration=1            #ITERATION COUNTER
    dx=0.0001            #STEP SIZE FOR FINITE DIFFERENCE
    max_iter=5000        #MAX NUMBER OF ITERATION
    tol=10**-10            #EXIT AFTER CHANGE IN F IS LESS THAN THIS 
    NDIM=len(xi)        #DIMENSION OF OPTIIZATION PROBLEM

    #OPTIMIZATION LOOP
    while(iteration<=max_iter):

        #-------------------------
        #DATASET PARITION BASED ON TRAINING PARADIGM
        #-------------------------
        if(PARADIGM=='batch'):
            if(iteration==1): index_2_use=train_idx
            if(iteration>1):  epoch+=1
        else:
            print("REQUESTED PARADIGM NOT CODED")

        #-------------------------
        #NUMERICALLY COMPUTE GRADIENT 
        #-------------------------
        df_dx=np.zeros(NDIM);    #INITIALIZE GRADIENT VECTOR
        for i in range(0,NDIM):    #LOOP OVER DIMENSIONS

            dX=np.zeros(NDIM);  #INITIALIZE STEP ARRAY
            dX[i]=dx;             #TAKE SET ALONG ith DIMENSION
            xm1=xi-dX;             #STEP BACK
            xp1=xi+dX;             #STEP FORWARD 

            #CENTRAL FINITE DIFF
            grad_i=(f(xp1,index_2_use)-f(xm1,index_2_use))/dx/2

            # UPDATE GRADIENT VECTOR 
            df_dx[i]=grad_i 
            
        #TAKE A OPTIMIZER STEP
        if(algo=="GD"):  xip1=xi-LR*df_dx 
        if(algo=="MOM"): print("REQUESTED ALGORITHM NOT CODED")

        #REPORT AND SAVE DATA FOR PLOTTING
        if(iteration%1==0):
            predict(xi)    #MAKE PREDICTION FOR CURRENT PARAMETERIZATION
            print(iteration,"    ",epoch,"    ",MSE_T,"    ",MSE_V) 

            #UPDATE
            epochs.append(epoch); 
            loss_train.append(MSE_T);  loss_val.append(MSE_V);

            #STOPPING CRITERION (df=change in objective function)
            df=np.absolute(f(xip1,index_2_use)-f(xi,index_2_use))
            if(df<tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break

        xi=xip1
        iteration=iteration+1

    return xi


#------------------------
#FIT MODEL
#------------------------

po=np.random.uniform(2,1.,size=NFIT)

p_final=minimizer(loss,po)        
print("OPTIMAL PARAM:",p_final)
predict(p_final)

#------------------------
#GENERATE PLOTS
#------------------------

#PLOT TRAINING AND VALIDATION LOSS HISTORY
def plot_0():
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_train, 'o', label='Training loss')
    ax.plot(epochs, loss_val, 'o', label='Validation loss')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.legend()
    plt.show()

def plot_1(xla='y_data',yla='y_predict'):
    fig, ax = plt.subplots()
    ax.plot(Y[train_idx]  , YPRED_T,'*', label='Training') 
    ax.plot(Y[val_idx]    , YPRED_V,'*', label='Validation') 
    ax.plot(Y[test_idx]    , YPRED_TEST,'*', label='Test') 
    plt.xlabel(xla, fontsize=18);    plt.ylabel(yla, fontsize=18);     plt.legend()
    plt.show()
    
if(IPLOT):

    plot_0()
    plot_1()

