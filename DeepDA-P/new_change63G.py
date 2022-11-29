# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:22:28 2022

@author: wensh
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
import xlrd
import json
import numpy as np

import my_get_body as mygetbody
import my_plot as myplot
import my_save_dataset as mysavedataset
import my_tree as mytree
import new_window as mywindow
import new_plot as myplot
def save_dataset(dataset_name,num,datax,datay):
    path='../'+dataset_name+'/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 



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
def change_one_people(h):   
    columns = ["ID","Sex","Status","Age","updrs_3_17a_off","updrs_3_17b_off"]
    fpath='../63dataset_original/'
    df=pd.read_csv(fpath+'label.csv',header=0,names=columns)
    fnum=df["ID"].values
    
    #print("ID",fnum)
    d={}
    d2={}
    global cnt
    X_train,Y_train,X_test,Y_test=[],[],[],[] 
    n=str(fnum[h])
    n = n.zfill(3)
    
   
    path='../'+'63dataset_original/'+n
    filenames = ['rh','lh']
    labelnames=['updrs_3_17a_off','updrs_3_17b_off']
    
    slabel=[]
    for i in range(len(filenames)):
        slabel.append(df[labelnames[i]].values)
    print("SLABEL:",slabel)
    columns=['activity']
    df=pd.read_csv("../63dataset_original/"+'activity.txt', sep='\t',header=None,names=columns)
    filekind=df['activity'].values
    #print("SS:",filekind)
    A_begin,A_end,A_hand=[],[],[]
    columns=["EventType",	"Start_Timestamp",	"Stop_Timestamp","Value"]
    df=pd.read_csv(path+'/'+"AnnotID"+n+".csv",header=0,names=columns) 
    activity_name=df['EventType'].values
    hand=df['Value'].values
    #print("HAND",hand)
    #print(df['Stop_Timestamp'].values)
    for k in range(0,len(activity_name)):
        num=activity_name[k].rfind('-')
        if num==-1:
            continue
        name=activity_name[k][num+2:]   
        #print("S:",name,len(activity_name),k)
        for i in range(len(filekind)):
                #print("TTT:",name)
            #if name==filekind[i]and not (df['Value'].values[k+2]=='OFF' or df['Value'].values[k+1]=='OFF' or df['Value'].values[k]=='OFF'):
            if name==filekind[i]:
                print("TTT:",name)
                # if df['Start_Timestamp'].values[k]==df['Stop_Timestamp'].values[k] :
                    #continue
                A_begin.append(df['Start_Timestamp'].values[k])
                A_end.append(df['Stop_Timestamp'].values[k])
                if hand[k]=='RIGHT':
                    A_hand.append(1)
                else :
                    A_hand.append(0)
    for i in range(len(A_begin)):
        print(A_begin[i],A_end[i],A_hand[i])
    #continue
    p_cnt=0
    for k in range(len(filenames)):#2个传感器
        flabel=slabel[k]
        print("YES",h,n,flabel[h])
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
      
        tmp_cnt=0
        #写的好差
        for t in range(len(A_begin)):
            #if A_hand[t]!=k:
                #continue
            segments,labels,pre_label=[],[],[]
            begin=0
            while(TI[begin]<A_begin[t]):
                begin+=1
            end=begin
            while(TI[end]<=A_end[t]):
                end+=1
            if end==begin:
                continue
            end=max(end,begin+128)
            xs = X[begin:end]
            ys = Y[begin:end]
            zs = Z[begin:end]
            print("BEGIN:",begin,end,TI[begin])
            #myplot.my_plot(xs,ys,zs,len(xs))
            xs=correct_HZ(xs, 31.5, 50)
            ys=correct_HZ(ys, 31.5, 50)
            zs=correct_HZ(zs, 31.5, 50)
            myplot.my_plot(xs, ys, zs,str(flabel[h])+'/'+str(fnum[h])+filenames[k]+str(A_begin[t]))
            m=[xs,ys,zs]
            m=np.asarray(m, dtype= np.float32)
            m=m.T
            segments.append(m)
            labels.append(flabel[h])
           
            cnt=cnt+1    
            save_dataset('63dataset',cnt,segments,labels)
   
def change_dataset():
    
    global cnt
    cnt=0
    for h in range(0,34):
        #if h==14:
        change_one_people(h)

change_dataset()