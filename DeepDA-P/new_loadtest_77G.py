# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:32:31 2021
@author: wensh
"""
import pandas as pd
import numpy as np
import math
import os

import my_tree as mytree
import my_save_dataset as mysavedataset
import new_window as mywindow

cnt=0
def load_one_people(A,B,da,h=50,N_TIME_STEPS=128,step=128):    
    n=str(h)
    n = n.zfill(3)
    path='../77Gdataset/'+'T'+n+'_'+A[h-1]
    filenames = os.listdir(path) 
    global cnt
    Y_train,X_train=[],[]
    for k in range(len(filenames)):
        if da.get(filenames[k],-1)==-1:
            continue
        columns=['x-axis', 'y-axis', 'z-axis','I1','x2-axis', 'y2-axis', 'z2-axis','I2','I3']
        df=pd.read_csv(path+'/'+filenames[k]+'/kinect_accelerometer.tsv', sep='\t',header=None,names=columns)
        X=df['x-axis'].values[::20]
        Y=df['y-axis'].values[::20]
        Z=df['z-axis'].values[::20]
        X=X/4000*98
        Y=Y/4000*98
        Z=Z/4000*98 
        
        cnt,segments,labels=mywindow.slide_window('77G',cnt,X,Y,Z,B[h-1],N_TIME_STEPS,step)
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
    dl={}
    columns=['num','Left','Right']
    df=pd.read_csv('../label.csv', header=None,names=columns)
    filetype=df['num'].values
    #测的是左手还是右手
    A,B=[],[]
    for i in range(len(filetype)):
        if filetype[i]=='Left':
            A.append('Left')
        else :
            A.append('Right')
        B.append(df[A[i]].values[i])
        #print(filetype[i],A[i],B[i])
    da={}
    columns=['activity']
    df=pd.read_csv('activity.txt', sep='\t',header=None,names=columns)
    filekind=df['activity'].values
    for k in range(len(filekind)):
        da[filekind[k]]=1
    X_train,Y_train =[],[]
    for h in range(1,56):
        X_tmp,Y_tmp=load_one_people(A,B,da,h)
        for i in range(len(X_tmp)):
            X_train.append(X_tmp[i])
            Y_train.append(Y_tmp[i])
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    print("check_shape_77G:",X_train.shape,  Y_train.shape)
    mywindow.cnt_class(Y_train)
    return X_train,Y_train
    
if __name__ == '__main__':
    X_train,Y_train=load_dataset()    

        