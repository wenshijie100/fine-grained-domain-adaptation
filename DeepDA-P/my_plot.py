# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:51:06 2021

@author: wensh
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot1(data):
    plt.figure(2)
    for i in range(data.shape[1]):
        plt.plot(data[:,i])
    plt.show()
    
color=['b','orange','g','y','c']
def plot_combine(datax,datay,y_mode=1,num=0):
    plt.figure(num)
    L=datax.shape[1]
    lx = np.linspace(0,L*datax.shape[0],L*datax.shape[0]+1)
    #print(datax.shape,lx)
    for k in range(datax.shape[0]):
        for i in range(datax.shape[2]):
            X=datax[k,:,i]
            t=lx[k*L:(k+1)*L]
            #print("LL",len(t),len(X))
            if datay[k]==y_mode:
                plt.plot(lx[k*L:(k+1)*L],X,color[i])
            else :
                plt.plot(lx[k*L:(k+1)*L],X,'k')
        #plt.xlim(0, 8)
    plt.show()
    
def plot2(X,Y):
    plt.figure(figsize=(12,20), dpi=150)    
    #print("XXXXXXXXX",X.shape,Y.shape,X.shape[1])
       
    for i in range(X.shape[1]):
        plt.subplot(X.shape[1], 1, i+1) 
        t=X[:,i]
        #print(t)
        plt.ylabel(Y, size=16)
        plt.ylim(-15, 15)
        plt.plot(X[:,i]) 

def save_plot(X,name):
    plt.figure(figsize=(30,30))
    for i in range(3):
        plt.subplot(X.shape[1], 1, i+1) 
        t=X[:,i]
        #print(t)
        #plt.ylabel(name, size=16)
        #plt.ylim(0, 1)
        plt.plot(X[:,i]) 
    fname=name+".jpg"
    plt.savefig('errors/'+fname)
    plt.close()

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
    
#my_plot_63()