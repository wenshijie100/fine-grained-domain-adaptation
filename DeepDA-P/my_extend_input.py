# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 23:03:39 2021

@author: wensh
"""
from pandas import Series
import numpy as np

def get_cor(c,d): 
    s1=Series(c) #转为series类型
    s2=Series(d)
    corr=s1.corr(s2)+s2.corr(s1) #计算相关系数
    corr=corr/2
    #print(corr)
    corr=np.array(corr)
    return corr


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
    #print("NEW1:",X.shape,new_X.shape,X.shape[2])
    return new_X

def get_input2(X):
    new_X=[]
   # print("X:",X.shape[2],int(X.shape[2]/3))
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
    #print("NEW2:",X.shape,new_X.shape,X.shape[2])
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
    #print("NEW3:",X.shape,new_X.shape,X.shape[2])
    return new_X

def extend_input(X):
    t1=get_input1(X)
    t2=get_input2(t1)
    t3=get_input3(t1)
    
    new_X=np.concatenate((t1,t2,t3),axis=2)
    #print("NEW4:",X.shape,new_X.shape)
    return new_X
    
    new_X=np.concatenate((t1,t2,t3),axis=2)
        