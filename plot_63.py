# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:07:06 2021

@author: wensh
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xlrd
import json
import numpy as np
import my_get_body as mygetbody
import my_plot as myplot
import csv
def getFlist(file_dir):
    fname_divide=[]
    ct=0
    for root, dirs, files in os.walk(file_dir):
        #print('root_dir:', root)
        #print('sub_dirs:', dirs)
        #print('files:', files)
        
        for j in range(len(files)):
            ct=ct+1
            ri = files[j].rindex('.')
            fname_divide.append(files[j][:ri])            
    print("CT:",ct)     
    #print(len(fname_divide))
    #print(fname_divide)     
    return fname_divide

def load_dataset():   
    columns = ["ID","Sex","Status","Age","updrs_3_17a_off","updrs_3_17b_off"]
    fpath='../63dataset/'
    df=pd.read_csv(fpath+'label.csv',header=None,names=columns)
    fnum=df["ID"].values
    flabel=df["updrs_3_17a_off"].values
    print("ID",fnum)
    d={}
    d2={}
    
                  
    
    X_train,Y_train,X_test,Y_test=[],[],[],[] 
    for h in range(len(fnum)):
        n=str(h)
        n = n.zfill(3)
        path='../'+'63dataset/'+n
        filenames = os.listdir(path)
        for k in range(len(filenames)):
        
            columns=['time','x-axis', 'y-axis', 'z-axis']
            df=pd.read_csv('../63dataset/007/rh_ID007Accel.csv',header=0,names=columns)
            X=df['x-axis'].values
            Y=df['y-axis'].values
            Z=df['z-axis'].values
            print(X,len(X))
            plt.figure(figsize=(12,20), dpi=150)
            X=X*9.8
            Y=Y*9.8
            Z=Z*9.8
            start,end=1000,5000
            mx=X[start:end].mean()
            my=Y[start:end].mean()
            mz=Z[start:end].mean()
            print(mz)
            X=X-mx
            Y=Y-my
            Z=Z-mz
            '''
            plt.figure(100)
            plt.plot(X)
            plt.plot(Y)
            plt.plot(Z)
            plt.show()
            '''
            N_TIME_STEPS=32
            step=32
            segments,labels=[],[]
            for i in range(0, len(X) - N_TIME_STEPS+1, step):
                    xs = X[i:i+N_TIME_STEPS]
                    ys = Y[i:i+N_TIME_STEPS]
                    zs = Z[i:i+N_TIME_STEPS]
                    '''
                    xs1 = X1[i:i+N_TIME_STEPS]
                    ys1 = Y1[i:i+N_TIME_STEPS]
                    zs1 = Z1[i:i+N_TIME_STEPS]
                    
                    xs2 = X2[i:i+N_TIME_STEPS]
                    ys2 = Y2[i:i+N_TIME_STEPS]
                    zs2 = Z2[i:i+N_TIME_STEPS]
                    '''
                    m=[xs,ys,zs]
                       
                    m=np.asarray(m, dtype= np.float32)
                    m=m.T
                    segments.append(m)
                    labels.append(flabel[h])
    X_train=segments
    Y_train=labels
    
    X_train=np.asarray(X_train,dtype = np.float32)
    #X_test=np.asarray(X_test,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    #Y_test=np.asarray(Y_test,dtype = np.float32)
    print("check_shape:",X_train.shape,  Y_train.shape)
    return X_train,  Y_train  



def my_plot_63():
    columns=['time','x-axis', 'y-axis', 'z-axis']
    df=pd.read_csv('../63dataset/007/rh_ID007Accel.csv',header=0,names=columns)
    X=df['x-axis'].values
    Y=df['y-axis'].values
    Z=df['z-axis'].values
    print(X,len(X))
    plt.figure(figsize=(12,20), dpi=150)
    X=X*9.8
    Y=Y*9.8
    Z=Z*9.8
    start,end=1000,5000
    mx=X[start:end].mean()
    my=Y[start:end].mean()
    mz=Z[start:end].mean()
    print(mz)
    plt.plot(X[start:end]-mx)
    plt.plot(Y[start:end]-my)
    plt.plot(Z[start:end]-mz)
    plt.show()
    
my_plot_63()
load_dataset()