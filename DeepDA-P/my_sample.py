# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 23:15:07 2021

@author: wensh
"""
import numpy as np
import time

def cut_dataset(X,Y,cutlen=32):
    new_X,new_Y=[],[]
    for k in range(X.shape[0]):
        L=X.shape[1]
        N_TIME_STEPS=cutlen
        step=int(cutlen/2)
        #print("CCCCCCCCCC",L,step,N_TIME_STEPS,L- N_TIME_STEPS+1)
        for i in range(0, L- N_TIME_STEPS+1, step):
            xs = X[k][i:i+N_TIME_STEPS]
            #print("XS:",xs.shape)
            new_X.append(xs)
            new_Y.append(Y[k])
          
    new_X=np.asarray(new_X,dtype = np.float32)    
    new_Y=np.asarray(new_Y,dtype = np.float32)    
    return new_X,new_Y

def extend_train(X,Y):
    new_X,new_Y=[],[]
    X=np.array(X)
    Y=np.array(Y)
    print("TRAIN:",X.shape,Y.shape)
    for i in range(len(Y)):
        new_X.append(X[i])
        new_Y.append(Y[i])
        if Y[i]!=0:
            new_X.append(X[i])
            new_Y.append(Y[i])
            new_X.append(X[i])
            new_Y.append(Y[i])
    new_X=np.array(new_X)    
    new_Y=np.array(new_Y)
    print("TRAIN:",new_X.shape,new_Y.shape)    
    return new_X,new_Y    

def deduct_train(X,Y):
    new_X,new_Y=[],[]
    X=np.array(X)
    Y=np.array(Y)
    print("TRAIN:",X.shape,Y.shape)
    for i in range(len(Y)):
        if Y[i]==0:
            t = time.time()
            t=int(round(t * 1000000)) 
            if t%2!=0:
                continue
            new_X.append(X[i])
            new_Y.append(Y[i])
        else:
             new_X.append(X[i])
             new_Y.append(Y[i])
       
    new_X=np.array(new_X)    
    new_Y=np.array(new_Y)
    print("TRAIN:",new_X.shape,new_Y.shape)    
    return new_X,new_Y    