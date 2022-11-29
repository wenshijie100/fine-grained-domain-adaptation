# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:50:36 2021

@author: wensh
"""

import os
import tensorflow as tf
#os.environ['AUTOGRAPH_VERBOSITY'] = 1
import tensorflow.keras as keras
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, Dropout, GRU, LSTM,Flatten
from tensorflow.keras.layers import concatenate
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

# This is for showing the Tensorflow log
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pandas.core.frame import DataFrame
import json
import pandas as pd
import numpy as np  
from sklearn.impute import SimpleImputer as Imputer
#from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets   
from sklearn.model_selection import train_test_split,cross_val_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import joblib
from numpy import dstack
from pandas import read_csv
import numpy as np
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from sklearn.model_selection import train_test_split
from scipy import stats
import pandas as pd
import math 
from pandas import Series
from sklearn.metrics import confusion_matrix
import xlrd
import time
import datetime
from sklearn.decomposition import PCA  #导入主成分分析库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load a single file as a numpy array
from scipy import signal
import my_normlization as mynormlization
import my_save_dataset as mysavedataset
import my_init
import my_F1 as myF1
import random
dn=[304,421,130]
def make_own_train_test(percent=0.0):
    #print(123)
    K=int(percent*dn[0]+0.5)
    X_train,Y_train=[],[]
    X_test,Y_test,L_test=[],[],[]
    
    seq=[i+1 for i in range(dn[0])]
    random.shuffle(seq)
    for i in range(dn[0]):
        #datax,datay=mysavedataset.read_feature_own(seq[i])
        datax,datay=mysavedataset.read_dataset_own(seq[i])
        #print(len(datax),len(datay))
        if i<K:
            for j in range(len(datax)):
                X_test.append(datax[j])
                Y_test.append(datay[j])
            L_test.append(len(datax))
        else:
            for j in range(len(datax)):
                X_train.append(datax[j])
                Y_train.append(datay[j])   
    
    
    if percent==0:
        return X_train,Y_train
    if percent==1:
        return X_test,Y_test,L_test
    return X_train,Y_train,X_test,Y_test,L_test
def make_77G_train_test(percent=0.0):
    K=int(percent*dn[1]+0.5)
    X_train,Y_train=[],[]
    X_test,Y_test,L_test=[],[],[]
    
    seq=[i+1 for i in range(dn[1])]
    random.shuffle(seq)
    for i in range(dn[1]):
        datax,datay=mysavedataset.read_dataset_77G(seq[i])
        #print("SSSSADDD",datax.shape,datay.shape,X_train.shape,Y_train.shape)
        #datax,datay=my_init.quick_77G_dataset_one_people(seq[i])
        #print("SSSSADDD",datax.shape,datay.shape,len(datax))
        if i<K:
            for j in range(len(datax)):
                X_test.append(datax[j])
                Y_test.append(datay[j])
            L_test.append(len(datax))
        else:
            for j in range(len(datax)):
                X_train.append(datax[j])
                Y_train.append(datay[j])   
        
        
    
  
  
    if percent==0:
        return X_train,Y_train
    if percent==1:
        return X_test,Y_test,L_test
    #print("AAAAADDD",X_train.shape,Y_train.shape)
    return X_train,Y_train,X_test,Y_test,L_test
def make_63G_train_test(percent=0.0):
    K=int(percent*dn[2]+0.5)
    X_train,Y_train=[],[]
    X_test,Y_test,L_test=[],[],[]
    
    seq=[i+1 for i in range(dn[2])]
    random.shuffle(seq)
    for i in range(dn[2]):
        datax,datay=mysavedataset.read_dataset_63G(seq[i])
        if i<K:
            for j in range(len(datax)):
                X_test.append(datax[j])
                Y_test.append(datay[j])
            L_test.append(len(datax))
        else:
            for j in range(len(datax)):
                X_train.append(datax[j])
                Y_train.append(datay[j])   
    
    
    if percent==0:
        return X_train,Y_train
    if percent==1:
        return X_test,Y_test,L_test
    #print("AAAAADDD",X_train.shape,Y_train.shape)
    return X_train,Y_train,X_test,Y_test,L_test
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
def extend_train(X,Y):
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
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 100, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model=keras.models.Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def mul_evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0,3,64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
 	# head 1
	inputs1 = Input(shape=(n_timesteps,n_features))
	conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)
	# head 2
	inputs2 = Input(shape=(n_timesteps,n_features))
	conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)
	# head 3
	inputs3 = Input(shape=(n_timesteps,n_features))
	conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	# merge
	merged = concatenate([flat1, flat2, flat3])
	# interpretation
	dense1 = Dense(128, activation='relu')(merged)
	outputs = Dense(n_outputs, activation='softmax')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
	# save a plot of the model
	#plot_model(model, show_shapes=True, to_file='multichannel.png')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	class_weight = {
         0: 1.0,
         1: 4.0,
         2: 8.0,
         3: 4.0,
     }
	model.fit([trainX,trainX,trainX], trainy, class_weight=class_weight,epochs=epochs, batch_size=batch_size)
	loss,acc=model.evaluate([testX,testX,testX],testy, batch_size=32)
	
	Y_pred = model.predict([testX,testX,testX],batch_size = 32)
	
	#print(Y_pred)   
	Z_pred=[np.argmax(Y_pred[i]) for i in range(len(Y_pred))]
	print(Z_pred)
	Z_test=[np.argmax(testy[i]) for i in range(len(testy))]
	C=confusion_matrix(Z_test, Z_pred)
	print(C)
	print(acc)
	myF1.get_F1(C)

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment():
    #X_train,Y_train,X_test,Y_test ,L_test=make_77G_train_test(0.0)
    X_train,Y_train=make_77G_train_test(0.0)
    
    X_train2,Y_train2,X_test,Y_test ,L_test=make_own_train_test(0.5)
    X_train=X_train+X_train2
    Y_train=Y_train+Y_train2
    
    labels=Y_train+Y_test
    labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    Y_train=labels[:len(Y_train)]
    Y_test=labels[len(Y_train):]
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    
    #print("SHAPE:",trainX.shape, trainy.shape, testX.shape, testy.shape)
    trainX, trainy=extend_train(X_train,Y_train)
    testX,testy=X_test,Y_test
    trainX=extend_input(trainX)
    testX=extend_input(testX)
    trainy=np.array(trainy)
    testy=np.array(testy)
    print("SHAPE:",trainX.shape, trainy.shape, testX.shape, testy.shape)
    print("F",trainX.shape, trainy.shape)
    mul_evaluate_model(trainX, trainy, testX, testy)
    
    
def plot_variable_distributions(trainX):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	print(longX.shape)
	pyplot.figure()
	xaxis = None
	for i in range(longX.shape[1]):
		ax = pyplot.subplot(longX.shape[1], 1, i+1, sharex=xaxis)
		ax.set_xlim(-1, 1)
		if i == 0:
			xaxis = ax
		pyplot.hist(longX[:, i], bins=100)
	pyplot.show()



 
# run the experiment
run_experiment()
#plot_data()
