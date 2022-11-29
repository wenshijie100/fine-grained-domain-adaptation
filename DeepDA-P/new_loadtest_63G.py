# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 13:50:27 2021=
@author: wensh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xlrd
import json
import numpy as np
import csv

import my_get_body as mygetbody
import my_save_dataset as mysavedataset
import new_window as mywindow
import new_plot as myplot
def read_dataset(dataset_name,num):
    path='../'+dataset_name+'/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay
cnt=0
def load_one_people(h,N_TIME_STEPS=128,step=128):   
    global cnt
    datax,datay=read_dataset('63dataset',h)
    datay=datay[0]
    X=datax[0,:,0]
    Y=datax[0,:,1]
    Z=datax[0,:,2]

    X_train,Y_train=[],[]
    cnt,segments,labels=mywindow.slide_window('63G',cnt,X,Y,Z,datay,N_TIME_STEPS,step)
   
    if len(segments)==0:
        return [],[]
    if datay==1 :
        myplot.my_plot(X,Y,Z,str(datay)+'/'+str(h))
    
    for i in range(len(segments)):
        X_train.append(segments[i])
        Y_train.append(labels[i])
       
    labels=np.array(labels)
    X_train=np.asarray(X_train,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    
    return X_train,  Y_train 
def load_dataset():
    global cnt
    cnt=0
    X_train,Y_train =[],[]
    for h in range(1,440):
        X_tmp,Y_tmp=load_one_people(h)
        for i in range(len(X_tmp)):
            X_train.append(X_tmp[i])
            Y_train.append(Y_tmp[i])
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    print("check_shape_63G:",X_train.shape,  Y_train.shape) 
    mywindow.cnt_class(Y_train)
    return X_train,Y_train



def my_plot_63():
    columns=['time','x-axis', 'y-axis', 'z-axis']
    df=pd.read_csv('../63dataset_original/005/rl_ID005Accel.csv',header=0,names=columns)
    X=df['x-axis'].values
    Y=df['y-axis'].values
    Z=df['z-axis'].values
    print(X,len(X))
    plt.figure(figsize=(12,20), dpi=150)
    X=X*9.8
    Y=Y*9.8
    Z=Z*9.8
    start,end=0,len(X)
    mx=X[start:end].mean()
    my=Y[start:end].mean()
    mz=Z[start:end].mean()
    #print(mz)
    plt.plot(X[start:end]-mx)
    plt.plot(Y[start:end]-my)
    plt.plot(Z[start:end]-mz)
    plt.show()
my_plot_63()
#load_dataset()