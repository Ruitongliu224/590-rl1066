#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize


#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='/Users/ruitongliu/Desktop/590/weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
model_type="logistic"
xcol=1; ycol=2;  
NFIT=4

#HYPER-PARAM
OPT_ALGO='CG'


# In[2]:



#------------------------
#DATA CLASS
#------------------------

class DataClass:

	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			with open(FILE_NAME) as f:
				self.input = json.load(f)


			X=[];
			for key in self.input.keys():
				if(key in DATA_KEYS):
					X.append(self.input[key])

			self.X=np.transpose(np.array(X))

			self.XMEAN=np.mean(self.X,axis=0)
			self.XSTD=np.std(self.X,axis=0)

		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED");

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:", self.X.shape)
		print("X examples")
		print("X means:",np.mean(self.X,axis=0))
		print("X stds:",np.std(self.X,axis=0))

		for i in range(0,self.X.shape[1]):
			print("X column ",i,": ",self.X[0:5,i])

	def partition(self,f_train=0.8, f_val=0.15, f_test=0.05):

		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		rand_indices = np.random.permutation(self.X.shape[0])
		CUT1=int(f_train*self.X.shape[0]); 
		CUT2=int((f_train+f_val)*self.X.shape[0]); 
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]

	def plot_xy(self,col1=1,col2=2,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			FS=18
			ax.plot(self.X[:,col1], self.X[:,col2],'o') 
			plt.xlabel(xla, fontsize=FS)
			plt.ylabel(yla, fontsize=FS)
			plt.show()

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD


# In[3]:


#------------------------
#MAIN 
#------------------------

data=DataClass(INPUT_FILE)

data.report()
data.partition()
data.normalize()
data.report()

#data.plot_xy(1,2,'age (years)','weight (lb)')
#data.plot_xy(2,0,'weight (lb)','is_adult')


# In[4]:


#------------------------
#DEFINE MODEL
#------------------------

def model(x,p):
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

def unnorm(x,col): 
	return data.XSTD[col]*x+data.XMEAN[col] 


#------------------------
#DEFINE LOSS FUNCTION

def loss_training(p):
	#TRAINING LOSS
    yp=model(xtrain,p)
    loss_training=(np.mean((yp-ytrain)**2.0))  

    return loss_training

    
def loss_validation(p):
	#VALIDATION LOSS
	yp=model(xval,p) 
	loss_validation=(np.mean((yp-yval)**2.0))

	return loss_validation


# In[5]:


#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
po=np.random.uniform(0.5,1.,size=NFIT)

#TRAINING
xtrain=data.X[:,xcol][data.train_idx]	
ytrain=data.X[:,ycol][data.train_idx] 
#VALIDATION
xval=data.X[:,xcol][data.val_idx]	
yval=data.X[:,ycol][data.val_idx] 
#TEST
xtest=data.X[:,xcol][data.test_idx]	
ytest=data.X[:,ycol][data.test_idx] 


# In[6]:


#MODEL USING OPTIMIZER
#def optimizer(objective, algo=‘GD’, LR=0.001, method=‘batch’)
PARAMETERS=[]; iterations=[]; iterations2=[];loss_train=[];loss_val=[]
def optimizer(objective, algo="GD", LR=0.001, method="batch"):
    
    #PARAMETRIC
    xmin=-50; xmax=50;  
    NDIM=4
    xi=np.random.uniform(xmin,xmax,NDIM)
    global PARAMETERS, iterations, iterations2, loss_train, loss_val
    dx=0.01
    LR=0.001
    t=0
    tmax=10000000
    tol=10**-7
    mu=0.5
    m=0
    v=0

    
    print("INITAL GUESS: ",xi)
    
    if (objective == loss_training):    
        while(t<=tmax):
        	t=t+1
        
        	#NUMERICALLY COMPUTE GRADIENT 
        	df_dx=np.zeros(NDIM)
        	for i in range(0,NDIM):
        		dX=np.zeros(NDIM);
        		dX[i]=dx; 
        		xm1=xi-dX; 
        		df_dx[i]=(objective(xi)-objective(xm1))/dx      
            
        	if (algo=="GD"):
                	xip1=xi-LR*df_dx
        	
        	if (algo=="GD and momentum"):                    
                    v = mu * v - LR*df_dx
                    xip1 = xi + v
      
       
        	if(t%10==0):              
                    df=np.mean(np.absolute(objective(xip1)-objective(xi)))
                    PARAMETERS.append(xi)
                    iterations.append(t)
                    loss_train.append(objective(xi))
                
                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")    
                        break
        
        	xi=xip1
    
    if (objective == loss_validation):    
        while(t<=tmax):
        	t=t+1
        
        	#NUMERICALLY COMPUTE GRADIENT 
        	df_dx=np.zeros(NDIM)
        	for i in range(0,NDIM):
        		dX=np.zeros(NDIM);
        		dX[i]=dx; 
        		xm1=xi-dX;
        		df_dx[i]=(objective(xi)-objective(xm1))/dx
                
        	if (algo=="GD"):
                	xip1=xi-LR*df_dx
        	
        	if (algo=="GD and momentum"):                    
                    v = mu * v - LR*df_dx
                    xip1 = xi + v   
        
        	if(t%10==0):              
                    df=np.mean(np.absolute(objective(xip1)-objective(xi)))
                    iterations2.append(t)
                    loss_val.append(objective(xi))
                
                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")    
                        break
        
        	xi=xip1 
            
    return PARAMETERS[-1]


# In[7]:


Algo = ["GD", "GD and momentum"]

for i in Algo:
    
    popt = optimizer(loss_training,algo=i)
    optimizer(loss_validation,algo=i)  
    
    xm=np.array(sorted(xtrain))
    yp=np.array(model(xm,popt))
    
    #FUNCTION PLOTS
    if(IPLOT):
    	fig, ax = plt.subplots()
    	ax.plot(unnorm(xtrain,xcol), unnorm(ytrain,ycol), 'o', label='Training set')
    	ax.plot(unnorm(xval,xcol), unnorm(yval,ycol), 'x', label='Validation set')
    	ax.plot(unnorm(xtest,xcol), unnorm(ytest,ycol), '*', label='Test set')
    	ax.plot(unnorm(xm,xcol),unnorm(yp,ycol), '-', label='Model')
    	plt.title("Comparison Plot: %s" %i)
    	plt.xlabel('x', fontsize=18)
    	plt.ylabel('y', fontsize=18)
    	plt.legend()
    	plt.show()


# In[8]:


#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xtrain,popt), ytrain, 'o', label='Training set')
	ax.plot(model(xval,popt), yval, 'o', label='Validation set')
	ax.plot(ytrain, ytrain, '-', label='y_predicted=y_data')
	plt.title("Parity Plot: %s" %i)    
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()


# In[9]:


# TRAINING & VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	#iterations,loss_train,loss_val
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations2, loss_val, 'o', label='Validation loss')
	plt.title("Training & Validation Loss: %s" %i)    
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()


# In[ ]:




