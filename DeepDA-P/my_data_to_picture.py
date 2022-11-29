# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:55:02 2021

@author: wensh
"""
import os
from numpy import mean
from numpy import std
from numpy import dstack

from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np  
from scipy import stats
import pandas as pd
import math 
import time
import datetime
from scipy import signal
import random

import my_normlization as mynormlization
import new_save_dataset as mysavedataset
import my_F1 as myF1
def run_finetune(train_dataset,test_dataset,tname="transfer+fintune"):
    #X_train,Y_train,X_test,Y_test ,L_test=make_77G_train_test(0.0)
    print("run my fintune code")

    X_train,Y_train,W_train,P_train,I_train,L_train=make_train_and_test(train_dataset,1.0)
    #X_test,Y_test,W_test,P_test,I_test,L_test=make_train_and_test(test_dataset,1.0)
   # X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2=X_test,Y_test,W_test,P_test,I_test,L_test
        
    X_test2,Y_test2,W_test2,P_test2,I_test2,L_test2,X_test,Y_test,W_test,P_test,I_test,L_test=make_train_and_test(test_dataset,0.7)
    X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2=make_train_and_test(test_dataset,1.0)
    #X_test,Y_test,W_test,P_test,I_test,L_test=X_test2,Y_test2,W_test2,P_test2,I_test2,L_test2
    print("TRAIN Length:",len(Y_train),len(L_train),"FINETUNE Length:",len(Y_test2),len(L_test2),"TEST Length:",len(Y_test),len(L_test))
    
    #print("LEGNTH:",len(X_train),len(X_train2),len(X_test),len(X_test2),len(L_test),len(L_test2))
    X_train=X_train2+X_train
    Y_train=Y_train2+Y_train
    W_train=W_train2+W_train
    P_train=P_train2+P_train
    I_train=I_train2+I_train
    L_train=L_train2+L_train
     
    X_test=X_test2+X_test
    Y_test=Y_test2+Y_test
    W_test=W_test2+W_test
    P_test=P_test2+P_test
    I_test=I_test2+I_test
    L_test=L_test2+L_test
    
    #print("TRAIN Length:",len(Y_train),len(L_train),"TEST Length:",len(Y_test),len(L_test))
    labels=Y_train+Y_test
    labels=[int(labels[i]) for i in range(len(labels))]
    #labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    Y_train=labels[:len(Y_train)]
    Y_test=labels[len(Y_train):]
    
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    W_train=np.array(W_train)
    P_train=np.array(P_train)
    I_train=np.array(I_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)
    W_test=np.array(W_test)
    P_test=np.array(P_test)
    I_test=np.array(I_test)

    X_data=np.concatenate((X_train,X_test),axis=0)
    X_data=(X_data-X_data.min())/(X_data.max()-X_data.min())
    X_train=X_data[:len(X_train)]
    X_test=X_data[len(X_train):]
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    #print("SHAPE:",trainX.shape, trainy.shape, testX.shape, testy.shape)
    trainX, trainY,trainW,trainP,trainI,trainL=X_train,Y_train,W_train,P_train,I_train,L_train
    testX,testY,testW,testP,testI,testL=X_test,Y_test,W_test,P_test,I_test,L_test
    
    trainY=np.array(trainY)
    testY=np.array(testY)
    #print("SHAPE:",trainX.shape, trainy.shape, testX.shape, testy.shape)
   
    trainX=trainX.transpose(0,2,1)
    testX=testX.transpose(0,2,1)
    length=len(Y_train2)
    leng=len(Y_test2)
    print("LL:",leng,length,len(testL),len(trainL))
    # A,B=shuf(A,B)
    # C,D=shuf(C,D)
    #print("ABC:",len(testL[len(L_test2):]),len(testL[:len(L_test2)]),max(testP[leng:]),max(testP[:leng]))
    return trainX[length:],trainY[length:],trainW[length:],trainP[length:],trainI[length:],trainL[len(L_train2):],\
    trainX[:length],trainY[:length],trainW[:length],trainP[:length],trainI[:length],trainL[:len(L_train2)],\
    testX[leng:],testY[leng:],testW[leng:],testP[leng:],testI[leng:],testL[len(L_test2):],\
    testX[:leng],testY[:leng],testW[:leng],testP[:leng],testI[:leng],testL[:len(L_test2)]

def run_experiment(train_dataset,test_dataset,tname="transfer"):
    #X_train,Y_train,X_test,Y_test ,L_test=make_77G_train_test(0.0)
    print("run my Parkinson code")

    X_train,Y_train,W_train,P_train,I_train,L_train=make_train_and_test(train_dataset,1.0)
    X_test,Y_test,W_test,P_test,I_test,L_test=make_train_and_test(test_dataset,1.0)
    X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2=X_test,Y_test,W_test,P_test,I_test,L_test
    if tname=="transfer+self":
        X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2,X_test,Y_test,W_test,P_test,I_test,L_test=make_train_and_test(test_dataset,0.7)
        X_train=X_train2+X_train
        Y_train=Y_train2+Y_train
        W_train=W_train2+W_train
        P_train=P_train2+P_train
        I_train=I_train2+I_train
        L_train=L_train2+L_train
        X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2=X_test,Y_test,W_test,P_test,I_test,L_test
        print("transfer+self")
    elif tname=="self":
        X_train,Y_train,W_train,P_train,I_train,L_train,X_test,Y_test,W_test,P_test,I_test,L_test=make_train_and_test(test_dataset,0.7)
        X_train2,Y_train2,W_train2,P_train2,I_train2,L_train2=X_train,Y_train,W_train,P_train,I_train,L_train
        print("self")
    elif tname=="transfer":
        print("transfer")
    else :
        print("ERROR")
    print("TRAIN Length:",len(Y_train),len(L_train),"TEST Length:",len(Y_test),len(L_test))
    
    X_train=X_train2+X_train
    Y_train=Y_train2+Y_train
    W_train=W_train2+W_train
    P_train=P_train2+P_train
    I_train=I_train2+I_train
    L_train=L_train2+L_train
    labels=Y_train+Y_test
    labels=[int(labels[i]) for i in range(len(labels))]
    #labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    Y_train=labels[:len(Y_train)]
    Y_test=labels[len(Y_train):]
    
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    W_train=np.array(W_train)
    P_train=np.array(P_train)
    I_train=np.array(I_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)
    W_test=np.array(W_test)
    P_test=np.array(P_test)
    I_test=np.array(I_test)

    X_data=np.concatenate((X_train,X_test),axis=0)
    X_data=(X_data-X_data.min())/(X_data.max()-X_data.min())
    X_train=X_data[:len(X_train)]
    X_test=X_data[len(X_train):]
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    #print("SHAPE:",trainX.shape, trainy.shape, testX.shape, testy.shape)
    trainX, trainY,trainW,trainP,trainI,trainL=X_train,Y_train,W_train,P_train,I_train,L_train
    testX,testY,testW,testP,testI,testL=X_test,Y_test,W_test,P_test,I_test,L_test
    
    trainY=np.array(trainY)
    testY=np.array(testY)
    #print("SHAPE:",trainX.shape, trainy.shape, testX.shape, testy.shape)
   
    trainX=trainX.transpose(0,2,1)
    testX=testX.transpose(0,2,1)
    length=len(Y_train2)
    # A,B=shuf(A,B)
    # C,D=shuf(C,D)
    
    return trainX[length:],trainY[length:],trainW[length:],trainP[length:],trainI[length:],trainL[len(L_train2):],\
    trainX[:length],trainY[:length],trainW[:length],trainP[:length],trainI[:length],trainL[:len(L_train2)],\
    testX,testY,testW,testP,testI,testL
   
def file_cnt(file_name,dataset_name):
    path='../'+file_name+'/'+dataset_name
    for root ,dirs ,files in os.walk(path):
        return int((len(files)/3)-1)

def make_train_and_test(dataset_name,percent=1.0):
    dn=file_cnt("dataset_weight",dataset_name)
    #print("P:",dn,type(dn))
    K=int(percent*dn+0.5)
    #print("DATASET:",dataset_name,"NUM:",dn)
    X_train,Y_train,W_train,P_train,I_train,L_train=[],[],[],[],[],[]
    X_test,Y_test,W_test,P_test,I_test,L_test=[],[],[],[],[],[]
    seq=[i+1 for i in range(dn)]
    cnt1,cnt2=0,0
    while(cnt1*cnt2==0):
        random.shuffle(seq)
        X_train,Y_train,W_train,P_train,I_train,L_train=[],[],[],[],[],[]
        X_test,Y_test,W_test,P_test,I_test,L_test=[],[],[],[],[],[]
        cnt1,cnt2=0,0
        for i in range(dn):
            datax,datay,dataw=mysavedataset.read_dataset_W(dataset_name,seq[i])
            #print(seq[i],len(datax),len(datay),len(dataw))
            if i<K:
                for j in range(len(datax)):
                    X_test.append(datax[j])
                    Y_test.append(datay[j])
                    W_test.append(dataw[j])
                    P_test.append(i)
                    I_test.append(j)
                    if datay[j]==3:
                        cnt1=cnt1+1
                L_test.append(len(datax))
            else:
                for j in range(len(datax)):
                    X_train.append(datax[j])
                    Y_train.append(datay[j])   
                    W_train.append(dataw[j])
                    P_train.append(i-K)
                    I_train.append(j)
                    if datay[j]==3:
                        cnt2=cnt2+1
                L_train.append(len(datax))
        if percent==0 or percent ==1:
            break
    if percent==0:
        return X_train,Y_train,W_train,P_train,I_train,L_train
    if percent==1:
        return X_test,Y_test,W_test,P_test,I_test,L_test
    #print("AAAAADDD",X_train.shape,Y_train.shape)

    return X_train,Y_train,W_train,P_train,I_train,L_train,X_test,Y_test,W_test,P_test,I_test,L_test

def get_body(data):
    newdata = np.copy(data)
    b, a = signal.butter(3, 0.95,'lowpass')
    newdata= signal.lfilter(b, a, data)  # data为要过滤的信号
    return newdata
def get_body2(data):
    newdata = np.copy(data)
    b, a = signal.butter(3, 0.05,'highpass')
    newdata= signal.lfilter(b, a, data)  # data为要过滤的信号
    return newdata
def get_grave2(data):
    newdata = np.copy(data)
    b, a = signal.butter(3, 0.05,'lowpass')
    newdata= signal.lfilter(b, a, data)  # data为要过滤的信号
    return newdata

def get_Jerk(data):
    t=[]
    #print(data.shape,data[0])
    for i in range (len(data)):
        for j in range(len(data[i])):
            if j==len(data[i])-1:
                t.append(t[-1])
            else :
                t.append(data[i][j+1]-data[i][j])
    t=np.array(t)
    t=t.reshape(data.shape[0],data.shape[1],1)
    return t

def get_avl(data):
    t=[]
    #print(data.shape,data[0])
    for i in range (len(data)):
        for j in range(len(data[i])):
            s=0
            for k in range(len(data[i][j])):
               s=s+np.abs(data[i][j][k])
            s=s/len(data[i][j])
            t.append(s)
    t=np.array(t)
    t=t.reshape(data.shape[0],data.shape[1],1)
    return t

def get_sqrt(data):
    t=[]
    #print(data.shape,data[0])
    for i in range (len(data)):
        for j in range(len(data[i])):
            s=0
            for k in range(len(data[i][j])):
               s=s+data[i][j][k]*data[i][j][k]
            s=np.sqrt(s)
            t.append(s)
    t=np.array(t)
    t=t.reshape(data.shape[0],data.shape[1],1)
    return t
def get_avl(data):
    t=[]
    #print(data.shape,data[0])
    for i in range (len(data)):
        for j in range(len(data[i])):
            s=0
            for k in range(len(data[i][j])):
               s=s+np.abs(data[i][j][k])
            s=s/len(data[i][j])
            t.append(s)
    t=np.array(t)
    t=t.reshape(data.shape[0],data.shape[1],1)
    return t
def get_input1(X):
    new_X=X
    for i in range(X.shape[2]):
        t=X[:,:,i]
        t=np.array(t)
        t=t.reshape(X.shape[0],X.shape[1],1)
        t=get_Jerk(t)
        #print(t.shape)
        new_X=np.concatenate((new_X,t),axis=2)
    new_X=np.array(new_X)
    print("NEW1:",X.shape,new_X.shape,X.shape[2])
    return new_X

def get_input2(X):
    new_X=[]
    print("X:",X.shape[2],int(X.shape[2]/3))
    for i in range(int(X.shape[2]/3)):
        t=X[:,:,i*3:(i+1)*3]
        t=np.array(t)
        t=t.reshape(X.shape[0],X.shape[1],3)
        t=get_sqrt(t)
        #print(t.shape)
        if i==0:
            new_X=t
        else :
            new_X=np.concatenate((new_X,t),axis=2)
    new_X=np.array(new_X)
    print("NEW2:",X.shape,new_X.shape,X.shape[2])
    return new_X

def get_input3(X):
    new_X=[]
    for i in range(int(X.shape[2]/3)):
        t=X[:,:,i*3:(i+1)*3]
        t=np.array(t)
        t=t.reshape(X.shape[0],X.shape[1],3)
        t=get_avl(t)
        #print(t.shape)
        if i==0:
            new_X=t
        else :
            new_X=np.concatenate((new_X,t),axis=2)
    new_X=np.array(new_X)
    print("NEW3:",X.shape,new_X.shape,X.shape[2])
    return new_X

def extend_input(X):
    t1=get_input1(X)
    t2=get_input2(t1)
    t3=get_input3(t1)
    
    new_X=np.concatenate((t1,t2,t3),axis=2)
    print("NEW4:",X.shape,new_X.shape)
    return new_X
def extend_train2(X,Y):
    new_X,new_Y=[],[]
    Y=[np.argmax(Y[i]) for i in range(len(Y))]
    X=np.array(X)
    Y=np.array(Y)
    print("TRAIN:",X.shape,Y.shape)
    for i in range(len(Y)):
        new_X.append(X[i])
        new_Y.append(Y[i])
        
        if Y[i]!=0:
            new_X.append(X[i])
            new_Y.append(Y[i])
            
    new_X=np.array(new_X)
    new_Y = np.asarray(pd.get_dummies(new_Y), dtype = np.float32)    
    new_Y=np.array(new_Y)
    print("TRAIN:",new_X.shape,new_Y.shape)    
    return new_X,new_Y    
def shuf(x,y):
    for i in range(len(x)):
            j = int(random.random() * (i + 1))
            if j<=len(x)-1:#交换
                x[i],x[j]=x[j],x[i]
                y[i],y[j]=y[j],y[i]
    return x,y
if __name__ == "__main__":
    # run the experiment
    run_experiment()
    #plot_data()
