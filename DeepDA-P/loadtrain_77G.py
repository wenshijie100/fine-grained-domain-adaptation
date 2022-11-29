# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:40:32 2021

@author: wensh
"""
import numpy as np
import xlrd
import pandas as pd
import os
import my_get_body as mygetbody
import my_plot as myplot
import my_clean_data as mycleandata
import my_normlization as mynormlization
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def get_var(datax):
    sum_var=0 
    for i in range(datax.shape[1]):
         t=datax[:,i]
         sum_var=sum_var+np.var(t)
    
    return sum_var


def get_amplitude(datax):#45,128,3
    A=[]
    for k in range(datax.shape[0]):
        for i in range(datax.shape[2]):
            t=datax[k,:,i]
            A.append(t.max()-t.min())
    #print('befor',A)
    A.sort()
    #print('after',A)
    A=np.array(A)     
    value=A.sum()/len(A)
    return value

def my_dbscan(dataX,K,n):
    y_pred = DBSCAN(eps = 2*(1.5+K/2), min_samples =n).fit_predict(dataX)
   
    y_pred=np.array(y_pred)
    #print("YY:",y_pred)
   
    return y_pred 

def load_dataset():    
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
    X_train,Y_train,X_test,Y_test=[],[],[],[] 
    for h in range(1,56):
        n=str(h)
        n = n.zfill(3)
        path='../'+'77Gdataset/'+'T'+n+'_'+A[h-1]
        filenames = os.listdir(path) 
        #print("FF:",filenames)
        #print(len(filekind))
        print("T:",h,B[h-1])
        for k in range(len(filenames)):
        
            if da.get(filenames[k],-1)==-1:
                continue
            columns=['x-axis', 'y-axis', 'z-axis','I1','x2-axis', 'y2-axis', 'z2-axis','I2','I3']
            df=pd.read_csv(path+'/'+filenames[k]+'/kinect_accelerometer.tsv', sep='\t',header=None,names=columns)
            #print(filenames[k])
            
            X=df['x-axis'].values[::20]
            Y=df['y-axis'].values[::20]
            Z=df['z-axis'].values[::20]
            X1=mygetbody.get_body2(X)
            Y1=mygetbody.get_body2(Y)
            Z1=mygetbody.get_body2(Z)    
            
            X2=mygetbody.get_grave2(X)
            Y2=mygetbody.get_grave2(Y)
            Z2=mygetbody.get_grave2(Z)
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
            N_TIME_STEPS=32
            step=16
            segments,labels=[],[]
            for i in range(0, len(X) - N_TIME_STEPS+1, step):
                    xs = X[i:i+N_TIME_STEPS]
                    ys = Y[i:i+N_TIME_STEPS]
                    zs = Z[i:i+N_TIME_STEPS]
                    
                    xs1 = X1[i:i+N_TIME_STEPS]
                    ys1 = Y1[i:i+N_TIME_STEPS]
                    zs1 = Z1[i:i+N_TIME_STEPS]
                    
                    xs2 = X2[i:i+N_TIME_STEPS]
                    ys2 = Y2[i:i+N_TIME_STEPS]
                    zs2 = Z2[i:i+N_TIME_STEPS]
                    m=[xs,ys,zs]
                       
                    m=np.asarray(m, dtype= np.float32)
                    m=m.T
                    segments.append(m)
                            #print("i",i,filename)
                    #labels.append(d[fname])
            segments=np.array(segments)
            #print(len(segments))
            isnoise=[i%1 for i in range(len(segments))]
            #plot_combine(segments,isnoise)
            #plot1(segments[19])
            amp=get_amplitude(segments)
            dataX=segments
            dataY=isnoise  
            tmp_data=np.concatenate((X.reshape(-1,1),Y.reshape(-1,1),Z.reshape(-1,1)), axis=1)
            var_sum=get_var(tmp_data)
            if B[h-1]!=0 and amp<5:
                print("var:",k,var_sum,amp,h,filenames[k],dataX.shape, tmp_data.shape)
                
            avl=[np.mean(X),np.mean(Y),np.mean(Z)]
            
            train_X,CC=mycleandata.get_tsfresh2(dataX,dataY,avl)
           
            train_X=mynormlization.normlization3(train_X)
            train_X=mynormlization.check_data(train_X)
            #print(train_X.shape,train_X[1])
            #print(train_X.shape)
            n=5+len(tmp_data)/5000
            if n>9:
                n=9
            isnoise=my_dbscan(train_X,np.sqrt(var_sum)/50,n)
            isnoise=isnoise+1
            tmp=np.bincount(isnoise)
            if len(tmp)==1:
                y_mode=-1
            else :
                y_mode=np.argmax(tmp[1:])+1
            #print("mode:",y_mode)
            #plot_combine(segments,isnoise,y_mode,k)
            cct=0
            for i in range(len(segments)):
                if isnoise[i]!=y_mode:
                    #print(i,isnoise[i],y_mode)
                    continue
                #if h%5==0:
                #    X_test.append(segments[i])
                #    Y_test.append(B[h-1])
                
                else :
                    cct=cct+1
                    X_train.append(segments[i])
                    Y_train.append(B[h-1])
            #print("legal:",cct)  
                
    X_train=np.asarray(X_train,dtype = np.float32)
    #X_test=np.asarray(X_test,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    #Y_test=np.asarray(Y_test,dtype = np.float32)
    print("check_shape:",X_train.shape,  Y_train.shape)
    return X_train,  Y_train  