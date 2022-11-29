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
import my_get_body as mygetbody
import my_plot as myplot
import csv
import my_save_dataset as mysavedataset
import my_tree as mytree
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
def check_time(num_begin,num_end,A_begin,A_end):
   
    if num_begin>=A_begin and num_end<=A_end:
        #print(num_begin,num_end,A_begin[i],A_end[i])
        return 1
    return 0
def correct_HZ(datax,data_HZ,standard_HZ):
    if data_HZ==standard_HZ:
        return datax
    num=[i for i in range(len(datax))]
    rate=float(data_HZ/standard_HZ)
    length=int(float(len(datax)/rate))
    new_num=[]
    tmp=0
    for i in range(length):
        new_num.append(tmp)
        tmp=tmp+rate
    new_datax=np.interp(new_num,num,datax)
    return new_datax
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
def load_one_people(h):   
    columns = ["ID","Sex","Status","Age","updrs_3_17a_off","updrs_3_17b_off"]
    fpath='../63dataset/'
    df=pd.read_csv(fpath+'label.csv',header=0,names=columns)
    fnum=df["ID"].values
    
    #print("ID",fnum)
    d={}
    d2={}
    
    global cnt
    
    X_train,Y_train,X_test,Y_test=[],[],[],[] 
    n=str(fnum[h])
    n = n.zfill(3)
    
    print("YES",h,n)
    path='../'+'63dataset/'+n
    filenames = ['rh','lh']
    labelnames=['updrs_3_17a_off','updrs_3_17b_off']
    
    slabel=[]
    for i in range(len(filenames)):
        slabel.append(df[labelnames[i]].values)
    
    #print(len(filenames),filenames)
    #break
        
    
    columns=['activity']
    df=pd.read_csv("../63dataset/"+'activity.txt', sep='\t',header=None,names=columns)
    filekind=df['activity'].values
    
    A_begin,A_end=[],[]
    columns=["EventType",	"Start_Timestamp",	"Stop_Timestamp","Value"]
    df=pd.read_csv(path+'/'+"AnnotID"+n+".csv",header=0,names=columns) 
    activity_name=df['EventType'].values
    on_off=df['Value'].values
    #print(df['Stop_Timestamp'].values)
    for k in range(0,len(activity_name)-2):
        
        num=activity_name[k].rfind('-')
        #print(activity_name[k])
        if num==-1:
            continue
        name=activity_name[k][num+2:]   
        #print("S:",name,len(activity_name),k)
        for i in range(len(filekind)):
            #print("TTT:",name)
            if name==filekind[i]and not (df['Value'].values[k+2]=='OFF' or df['Value'].values[k+1]=='OFF' or df['Value'].values[k]=='OFF'):
               # if df['Start_Timestamp'].values[k]==df['Stop_Timestamp'].values[k] :
                    #continue
                    A_begin.append(df['Start_Timestamp'].values[k])
                    A_end.append(df['Stop_Timestamp'].values[k])
    for i in range(len(A_begin)):
        print(A_begin[i],A_end[i])
    #continue
    p_cnt=0
    for k in range(len(filenames)):#2个传感器
        flabel=slabel[k]
        columns=['time','x-axis', 'y-axis', 'z-axis']
     
        try:    
            df=pd.read_csv(path+'/'+filenames[k]+"_ID"+n+"Accel.csv",header=0,names=columns)
        except:
            continue
        X=df['x-axis'].values
        Y=df['y-axis'].values
        Z=df['z-axis'].values
        TI=df['time'].values
        #print("LL",len(X))
        #plt.figure(figsize=(12,20), dpi=150)
        X=X*9.8
        Y=Y*9.8
        Z=Z*9.8
        X=correct_HZ(X, 30, 50)
        Y=correct_HZ(Y, 30, 50)
        Z=correct_HZ(Z, 30, 50)
        TI=correct_HZ(TI, 30, 50)
        
        #start,end=1000,5000
        mx=X.mean()
        my=Y.mean()
        mz=Z.mean()
        #print(mz)
        #X=X-mx
        #Y=Y-my
        #Z=Z-mz
        '''
        plt.figure(100)
        plt.plot(X)
        plt.plot(Y)
        plt.plot(Z)
        plt.show()
        '''
        N_TIME_STEPS=32
        step=32
        
        
        for t in range(len(A_begin)):
            segments,labels,pre_label=[],[],[]
            for i in range(0,len(TI) - N_TIME_STEPS+1, step):
                if check_time(TI[i],TI[i+step-1],A_begin[t],A_end[t])==0:
                    continue
                if TI[i]>A_end[t]:
                    break
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
                if check_illegal(xs,ys,zs,flabel[h])==0:
                    pre_label.append(0)
                else:
                    pre_label.append(1)
                
                #print(i,m)
                m=[xs,ys,zs]
                m=np.asarray(m, dtype= np.float32)
                m=m.T
                segments.append(m)
                labels.append(flabel[h])
                '''
                if p_cnt<10:
                    plt.figure(p_cnt)
                    plt.plot(xs)
                    plt.plot(ys)
                    plt.plot(zs)
                    plt.show()
                    p_cnt=p_cnt+1
                '''  
            if len(segments)==0:
                print("EOORR",h,k,t,A_begin[t],A_end[t],len(TI))
                continue
            segments,labels=mytree.K_tree(segments,labels,pre_label,h)

            if len(segments)>5:
                cnt=cnt+1
                mysavedataset.save_dataset_63G(cnt,segments,labels)
            else :
                continue
            segments,labels= mysavedataset.read_dataset_63G(cnt)
            for i in range(len(segments)):
                X_train.append(segments[i])
                Y_train.append(labels[i])
            
    
    X_train=np.asarray(X_train,dtype = np.float32)
    #X_test=np.asarray(X_test,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    #Y_test=np.asarray(Y_test,dtype = np.float32)
    print("check_shape:",X_train.shape,  Y_train.shape)
    return X_train,  Y_train  
def load_dataset():
    
    global cnt
    cnt=0
    X_train,Y_train =[],[]
    for h in range(0,34):
        #if h!=8:
            #continue
        X_tmp,Y_tmp=load_one_people(h)
        for i in range(len(X_tmp)):
            X_train.append(X_tmp[i])
            Y_train.append(Y_tmp[i])
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    print("check_shape_63G:",X_train.shape,  Y_train.shape,np.max(Y_train))
    return X_train,Y_train

def my_plot_63():
    columns=['time','x-axis', 'y-axis', 'z-axis']
    df=pd.read_csv('../63dataset/063/lh_ID063Accel.csv',header=0,names=columns)
    X=df['x-axis'].values
    Y=df['y-axis'].values
    Z=df['z-axis'].values
    print(X,len(X))
    plt.figure(figsize=(12,20), dpi=150)
    #X=X*9.8
    #Y=Y*9.8
    #Z=Z*9.8
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
load_dataset()