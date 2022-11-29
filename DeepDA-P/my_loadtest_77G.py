# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:32:31 2021

@author: wensh
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfresh as tsf
from pandas import Series
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import math
import os
import my_tree as mytree
import my_save_dataset as mysavedataset
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
    
def get_inr(data):
    t=[]
    lower_q=np.quantile(data,0.25,interpolation='lower')#下四分位数
    higher_q=np.quantile(data,0.75,interpolation='higher')#上四分位数
    int_r=higher_q-lower_q#四分位距
    #print(int_r)
    t.append(int_r)
    t=np.array(t)
    #print("T",t,t.shape)
    return t

def get_energy2(data):
    t=[]
    ts = pd.Series(data)  #数据x假设已经获取
    
    ae = tsf.feature_extraction.feature_calculators.abs_energy(ts)
    t.append(ae)
    '''
    ae=tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)
    t.append(ae)
    '''
    ae=tsf.feature_extraction.feature_calculators.count_above_mean(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.count_below_mean(ts)
    t.append(ae)
    '''
    ae = tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts)
    t.append(ae)
    '''
    ae=tsf.feature_extraction.feature_calculators.last_location_of_maximum(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.last_location_of_minimum(ts)
    t.append(ae)
    
    ae=tsf.feature_extraction.feature_calculators.longest_strike_above_mean(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.longest_strike_below_mean(ts)
    t.append(ae)
  
    #print(t)
    t=np.array(t)
    return t    
CC=10
def my_var(datax,avl):
    sum_var=0 
    for i in range (datax.shape[0]):
          sum_var=sum_var+pow(datax[i]-avl,2)
    t=[]
    t.append(sum_var)
    t=np.array(t)
    
    #print("T",t,t.shape)
    return t

def get_tsfresh2(datax,datay,avl):
    new_data=[]
    t_data=[]
      
    for i in range(datax.shape[0]):
        t=[1]
        t=np.array(t)
        for j in range(datax.shape[2]):# 30 channels
        
            X=datax[i,:,j]
            A=get_energy2(X)
            B=my_var(X,avl[j])
            C=get_inr(X)
            
            #print("T:",t.shape)
            t=np.concatenate((t,A,B,C,[i]), axis=0)
            #A=1 B=16
            
        t=t[1:]
        t=t.reshape(1,datax.shape[2]*CC)
        t=np.array(t)
        #print("T:",t.shape)
        #print("T:",t_data.shape)
        if i==0:
             t_data=t
        else :
            t_data=np.concatenate((t_data,t), axis=0)
        if(i%100==0):
            print(i)
    new_data=t_data
    new_data=np.array(new_data)
    #print(new_data.shape)
    return new_data      
def my_dbscan(dataX,K,n):
    y_pred = DBSCAN(eps = 2*(1.5+K/2), min_samples =n).fit_predict(dataX)
   
    y_pred=np.array(y_pred)
    #print("YY:",y_pred)
   
    return y_pred 


def check_illegal(xs,ys,zs,cs):
    if mytree.get_fangcha(xs)+mytree.get_fangcha(ys)+mytree.get_fangcha(zs)<0.5 and cs!=0:
        return 0
    if np.max(xs)>25 or np.max(ys)>25 or np.max(zs)>25:
        cnt_x_25,cnt_y_25,cnt_z_25=0,0,0
        for i in range(len(xs)):
            if xs[i]>15:
                cnt_x_25+=1
            if ys[i]>15:
                cnt_y_25+=1
            if zs[i]>15:
                cnt_z_25+=1
        if  cnt_x_25<=3 and  cnt_y_25<=3 and  cnt_z_25<=3:
            return 0
    if np.min(xs)<-25 or np.min(ys)<-25 or np.min(zs)<-25:
        cnt_x_25,cnt_y_25,cnt_z_25=0,0,0
        for i in range(len(xs)):
            if xs[i]<-15:
                cnt_x_25+=1
            if ys[i]<-15:
                cnt_y_25+=1
            if zs[i]<-15:
                cnt_z_25+=1
        if cnt_x_25<=3 and  cnt_y_25<=3 and  cnt_z_25<=3:
            return 0
    return 1
cnt=0
def load_one_people(A,B,da,h=50):    
      
    n=str(h)
    n = n.zfill(3)
    path='../77Gdataset/'+'T'+n+'_'+A[h-1]
    filenames = os.listdir(path) 
    #print("FF:",filenames)
    #print(len(filekind))
    #print("T:",h)
    global cnt
    Y_train,X_train=[],[]
    for k in range(len(filenames)):
        
        if da.get(filenames[k],-1)==-1:
            continue
        
        #print("OK")
        columns=['x-axis', 'y-axis', 'z-axis','I1','x2-axis', 'y2-axis', 'z2-axis','I2','I3']
        df=pd.read_csv(path+'/'+filenames[k]+'/kinect_accelerometer.tsv', sep='\t',header=None,names=columns)
        #print(filenames[k])
        
        X=df['x-axis'].values[::20]
        Y=df['y-axis'].values[::20]
        Z=df['z-axis'].values[::20]
        #X=X[364:396]
        #Y=Y[364:396]
        #Z=Z[364:396]
        X=X/4000*98
        Y=Y/4000*98
        Z=Z/4000*98
        #print("XXXXXXXXXXXX:",X.shape)
        '''
        plt.figure(100)
        plt.plot(X)
        plt.plot(Y)
        plt.plot(Z)
        plt.show()
        '''
        #print("FC:",mytree.get_fangcha(X),mytree.get_fangcha(X),mytree.get_fangcha(X))
        #if mytree.get_fangcha(X)+mytree.get_fangcha(Y)+mytree.get_fangcha(Z)<1 and B[h-1]!=0:
            #continue
        N_TIME_STEPS=128
        step=128
        segments,labels,pre_label=[],[],[]
        for i in range(0, len(X) - N_TIME_STEPS+1, step):
             if i<100:
                 continue
             else:
                xs = X[i:i+N_TIME_STEPS]
                ys = Y[i:i+N_TIME_STEPS]
                zs = Z[i:i+N_TIME_STEPS]
                
                   
                 
                if check_illegal(xs,ys,zs,B[h-1])==0:
                    pre_label.append(0)
                    continue
                else:
                    pre_label.append(1)
                    
                m=[xs,ys,zs]
                m=np.asarray(m, dtype= np.float32)
                m=m.T
                segments.append(m)
                        #print("i",i,filename)
                labels.append(B[h-1])
            
        #segments,labels=mytree.K_tree(segments,labels,pre_label,h)
        print("OMG",len(segments))
        if len(segments)>=2:
            cnt=cnt+1
            mysavedataset.save_dataset_77G(cnt,segments,labels)
        else :
            continue
        segments,labels= mysavedataset.read_dataset_77G(cnt)
        
        for i in range(len(segments)):
            X_train.append(segments[i])
            Y_train.append(labels[i])
        labels=np.array(labels)
        
        #if np.argmax(np.bincount(labels))!=B[h-1]:
            #print("ERROR:",h,np.bincount(labels),B[h-1],filenames[k])
   # X_train=normlization3(X_train)
   # X_train=check_data(X_train)
    X_train=np.asarray(X_train,dtype = np.float32)
    #X_test=np.asarray(X_test,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    #Y_test=np.asarray(Y_test,dtype = np.float32)
    #print("check_shape:",X_train.shape,  Y_train.shape)
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
        #print(B[i])
        #print(filetype[i],A[i],B[i])
    da={}
    columns=['activity']
    df=pd.read_csv('activity.txt', sep='\t',header=None,names=columns)
    filekind=df['activity'].values
    for k in range(len(filekind)):
        da[filekind[k]]=1
    
    X_train,Y_train =[],[]
    for h in range(1,56):
        #if h!=8:
            #continue
        X_tmp,Y_tmp=load_one_people(A,B,da,h)
        for i in range(len(X_tmp)):
            X_train.append(X_tmp[i])
            Y_train.append(Y_tmp[i])
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    print("check_shape_77G:",X_train.shape,  Y_train.shape,np.max(Y_train))
    return X_train,Y_train
    
if __name__ == '__main__':
    X_train,Y_train=load_dataset()    

        