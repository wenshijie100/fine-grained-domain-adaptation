# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:18:03 2021

@author: wensh
"""
import numpy as np
import pandas as pd
import tsfresh as tsf
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
    '''
    ae = tsf.feature_extraction.feature_calculators.abs_energy(ts)
    t.append(ae)
    '''
    ae=tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)
    t.append(ae)
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
    CC2=10
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
        t=t.reshape(1,datax.shape[2]*CC2)
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
    return new_data,CC2 