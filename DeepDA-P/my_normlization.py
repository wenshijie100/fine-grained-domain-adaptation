# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:25:02 2021

@author: wensh
"""
import math
import numpy as np
def normlization(data):
    data=np.array(data)
    for i in range(data.shape[2]):
        X=data[:,:,i]
        xmax=X.max()
        xmin=X.min()
        X=(X-xmin)/(xmax-xmin)
        data[:,:,i]=X
    return  data
def normlization2(data):
    data=np.array(data)
    for i in range(data.shape[1]):
        X=data[:,i]
        xmax=X.max()
        xmin=X.min()
        X=(X-xmin)/max(xmax-xmin,0.3)
        data[:,i]=X
    return  data
def normlization3(data):
    data=np.array(data)
    
    for i in range(data.shape[1]):
        X=data[:,i]
        xmax=X.max()
        xmin=X.min()
        #print(i,xmax-xmin)
        X=X*1.0
        #print("X:",X,xmax,xmin)
        X=(X-xmin)/max(xmax-xmin,0.3)
        #print("new_X:",X)    
        data[:,i]=X
        '''
        for j in range(len(data)):
            data[j][i]=X[i]
            print("DDDD:",data[j][i],X[i])
        '''
        #print("D:",data[:,i],X)
    return  data    

def check_data(datax):
    M=0
    for i in range(len(datax)):
       for j in range(len(datax[0])):
            t=datax[i][j]
            #print(i,j,t)
            if math.isinf(t) or math.isnan(t):
                #print("AAAAAAAAAAAA")
                datax[i][j]=M
    return datax

def normlization_train_test(data):
    data=np.array(data)
    
    for i in range(data.shape[1]):
        X=data[:,i]
        xmax=X.max()
        xmin=X.min()
        #print(i,xmax-xmin)
        X=X*1.0
        #print("X:",X,xmax,xmin)
        X=(X-xmin)/max(xmax-xmin,0.3)
        #print("new_X:",X)    
        data[:,i]=X
        '''
        for j in range(len(data)):
            data[j][i]=X[i]
            print("DDDD:",data[j][i],X[i])
        '''
        #print("D:",data[:,i],X)
    return  data    
