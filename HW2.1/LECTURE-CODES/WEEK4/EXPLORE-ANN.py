#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

#MAXIMUM FOR RANDOMIZATION
max_rand_wb=1

#NODES IN EACH LAYER [INPUT,H1,H2,OUTPUT]
layers		=	[2,3,2,1] 

#CALCULATE NUMBER OF FITTING PARAMETERS FOR SPECIFIED NN 
NFIT=0; 
for i in range(1,len(layers)):
	print("Nodes in layer-",i-1,"	=	",layers[i-1])  
	NFIT=NFIT+layers[i-1]*layers[i]+layers[i]

print("Nodes in layer-",i,"	=	",layers[i])  
print("NFIT			:	",NFIT)

#TAKES A LONG VECTOR W OF WEIGHTS AND BIAS AND RETURNS 
#WEIGHT AND BIAS SUBMATRICES
def extract_submatrices(WB):
	submatrices=[]; K=0
	for i in range(0,len(layers)-1):
		#FORM RELEVANT SUB MATRIX FOR LAYER-N
		Nrow=layers[i+1]; Ncol=layers[i] #+1
		w=np.array(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
		K=K+Nrow*Ncol; #print i,k0,K
		Nrow=layers[i+1]; Ncol=1; #+1
		b=np.transpose(np.array([WB[K:K+Nrow*Ncol]])) #unpack/ W 
		K=K+Nrow*Ncol; #print i,k0,K
		submatrices.append(w); submatrices.append(b)
		print(w.shape,b.shape)
	return submatrices

#RANDOM INITIAL GUESS
po=np.random.uniform(-max_rand_wb,max_rand_wb,size=NFIT)

print(po)
extract_submatrices(po)


