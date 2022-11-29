# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 23:13:11 2021

@author: wensh
"""
import numpy as np
import my_fft as FFT
sum_len=8
def get_ffeature(X):
    new_X=[]
    for i in range(X.shape[0]):
        new_data=[]
        for j in range(X.shape[2]):
            t=X[i,:,j]
            #print(t.shape)
            new_t=FFT.get_signal(20, 128, t,sum_len)
            if len(new_data)==0:
                new_data=new_t
            else :
                new_data=np.concatenate((new_data,new_t), axis=0)
            #print("T_shape:",new_t.shape,new_data.shape)
        new_data=new_data.reshape(1,sum_len*X.shape[2])
        if len(new_X)==0:
            new_X=new_data
        else :
            new_X=np.concatenate((new_X,new_data), axis=0)
    new_X=np.array(new_X)
   # print("X_shape:",new_X.shape)
    return new_X