# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 01:21:53 2022
100HZ
@author: wensh
"""
import matplotlib.pyplot as plt
import pickle
import new_window as mywindow
import numpy as np
def picklesave(obj,file):
    ff = open(file,'wb')
    pickle.dump(obj,ff)
    ff.close()

def pickleload(file):
    ff = open(file,'rb')
    obj = pickle.load(ff)
    ff.close()
    return obj

def load_dataset(N_TIME_STEPS=128,step=128):
    t=pickleload("../45dataset/tremor_dataset.pickle")
    cnt=0
    X_train,Y_train=[],[]
    for num in range(1,45):
        #print(len(t[num]['sessions'][1]))
        plt.figure(figsize=(12,20), dpi=150)
        #print(t[num]['annotation'])
        print("LEN:",len(t[num]['sessions']))
        label=max(t[num]['annotation'].values())
        print("LABEL:",label)
        for i in range(len(t[num]['sessions'])):
            X=t[num]['sessions'][i][::2,1]#100HZ
            Y=t[num]['sessions'][i][::2,2]
            Z=t[num]['sessions'][i][::2,3]
            label=min(label,3)
            cnt,segments,labels=mywindow.slide_window('45',cnt,X,Y,Z,label,N_TIME_STEPS,step)
            if len(segments)==0:
                continue
            for i in range(len(segments)):
                X_train.append(segments[i])
                Y_train.append(labels[i])  
    X_train=np.asarray(X_train,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    print("check_shape_45:",X_train.shape, Y_train.shape, Y_train.shape)
    mywindow.cnt_class(Y_train)
    return X_train,  Y_train 
def plot():
    t=pickleload("../45dataset/tremor_dataset.pickle")
    for num in range(1,45):
        #print(len(t[num]['sessions'][1]))
        plt.figure(figsize=(12,20), dpi=150)
        print(t[num]['annotation'])
        print("LEN:",len(t[num]['sessions']))
        X=t[num]['sessions'][1][:,1]
        Y=t[num]['sessions'][1][:,2]
        Z=t[num]['sessions'][1][:,3]
        #print(X)
        start,end=1000,5000
        mx=X[start:end].mean()
        my=Y[start:end].mean()
        mz=Z[start:end].mean()
        #print(mz)
        plt.plot(X[start:end]-mx)
        plt.plot(Y[start:end]-my)
        plt.plot(Z[start:end]-mz)
        plt.show()
if __name__ == '__main__':
    load_dataset()