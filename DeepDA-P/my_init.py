
"""
Created on Tue Oct 19 11:18:26 2021

@author: wensh
"""

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

from seglearn.feature_functions import all_features
from seglearn.transform import FeatureRep, SegmentXY
from seglearn.feature_functions import mean, var, std, minimum, maximum
from seglearn.pipe import Pype
from seglearn.transform import SegmentXY, last, middle, mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import random
from sklearn.model_selection import train_test_split
from scipy import stats
import pandas as pd
from sklearn.metrics import confusion_matrix
import xlrd
import time
import datetime
from sklearn.decomposition import PCA  #导入主成分分析库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from scipy import signal

import my_normlization as mynormlization
import my_feature1 as myfeature1
import my_feature2 as myfeature2
import my_extend_input as myextendinput
import my_save_dataset as mysavedataset
import my_tree as mytree
def init_own_dataset():
    for h in range(1,76):
        print(h)
        x_train0,y_train=mysavedataset.read_dataset_own(h)
        print("HSAPE:",x_train0.shape,y_train.shape)
        x_train0=myextendinput.extend_input(x_train0)
        
        n_axis=x_train0.shape[2]
        #x_train0=mynormlization.normlization(x_train0)
        x_train_o=x_train0
        y_train_o=y_train
        C,CC=myfeature1.get_tsfresh(x_train0,y_train)
        C=C.reshape(-1,n_axis*CC)
        E=myfeature2.get_ffeature(x_train0)
        x_train=np.concatenate((C,E), axis=1)
        #x_train=mynormlization.normlization2(x_train)   
        y_train=np.array(y_train)
        print("SHAPE:",x_train.shape,y_train.shape)
        mysavedataset.save_feature_own(h,x_train,y_train)
        
def init_77G_dataset():
    for h in range(1,423):
        print(h)
        x_train0,y_train=mysavedataset.read_dataset_77G(h)
        print("HSAPE:",x_train0.shape,y_train.shape)
        x_train0=myextendinput.extend_input(x_train0)
        
        n_axis=x_train0.shape[2]
        #x_train0=mynormlization.normlization(x_train0)
        x_train_o=x_train0
        y_train_o=y_train
        C,CC=myfeature1.get_tsfresh(x_train0,y_train)
        C=C.reshape(-1,n_axis*CC)
        E=myfeature2.get_ffeature(x_train0)
        x_train=np.concatenate((C,E), axis=1)
        #x_train=mynormlization.normlization2(x_train)   
        y_train=np.array(y_train)
        print("SHAPE:",x_train.shape,y_train.shape)
        mysavedataset.save_feature_77G(h,x_train,y_train)
        
def init_63G_dataset():
    for h in range(1,130):
        print(h)
        x_train0,y_train=mysavedataset.read_dataset_63G(h)
        print("HSAPE:",x_train0.shape,y_train.shape)
        x_train0=myextendinput.extend_input(x_train0)
        
        n_axis=x_train0.shape[2]
        #x_train0=mynormlization.normlization(x_train0)
        x_train_o=x_train0
        y_train_o=y_train
        C,CC=myfeature1.get_tsfresh(x_train0,y_train)
        C=C.reshape(-1,n_axis*CC)
        E=myfeature2.get_ffeature(x_train0)
        x_train=np.concatenate((C,E), axis=1)
        #x_train=mynormlization.normlization2(x_train)   
        y_train=np.array(y_train)
        print("SHAPE:",x_train.shape,y_train.shape)
        mysavedataset.save_feature_63G(h,x_train,y_train)

#####################################用于展示数据集

def quick_own_dataset_one_people(h):
    #print("OWN",h)
    datax,datay=[],[]
    
    x_train0,y_train=mysavedataset.read_dataset_own(h)
    #print("HSAPE:",x_train0.shape,y_train.shape)
    x_train,CC=mytree.get_tsfresh3(x_train0)
    y_train=np.array(y_train)
    #print("SHAPE:",x_train.shape,y_train.shape)
    #mysavedataset.save_feature_77G(h,x_train,y_train)
    for i in range (len(x_train)):
        datax.append(x_train[i])
        datay.append(y_train[i])
    datax=np.array(datax)
    datay=np.array(datay)
    return datax,datay
    
        
def quick_77G_dataset_one_people(h):
    #print("77G",h)
    datax,datay=[],[]
    
    x_train0,y_train=mysavedataset.read_dataset_77G(h)
    #print("HSAPE:",x_train0.shape,y_train.shape)
    x_train,CC=mytree.get_tsfresh3(x_train0)
    y_train=np.array(y_train)
    #print("SHAPE:",x_train.shape,y_train.shape)
    #mysavedataset.save_feature_77G(h,x_train,y_train)
    for i in range (len(x_train)):
        datax.append(x_train[i])
        datay.append(y_train[i])
    datax=np.array(datax)
    datay=np.array(datay)
    return datax,datay
def quick_own_dataset():
    print("OWN")
    datax,datay=[],[]
    for h in range(1,76):
        print(h)
        x_train0,y_train=mysavedataset.read_dataset_own(h)
        #print("HSAPE:",x_train0.shape,y_train.shape)
        x_train,CC=mytree.get_tsfresh3(x_train0)
        y_train=np.array(y_train)
        #print("SHAPE:",x_train.shape,y_train.shape)
        #mysavedataset.save_feature_77G(h,x_train,y_train)
        for i in range (len(x_train)):
            datax.append(x_train[i])
            datay.append(y_train[i])
    datax=np.array(datax)
    datay=np.array(datay)
    return datax,datay
    
        
def quick_77G_dataset():
    print("77G")
    datax,datay=[],[]
    for h in range(1,423):
        print(h)
        x_train0,y_train=mysavedataset.read_dataset_77G(h)
        #print("HSAPE:",x_train0.shape,y_train.shape)
        x_train,CC=mytree.get_tsfresh3(x_train0)
        y_train=np.array(y_train)
        #print("SHAPE:",x_train.shape,y_train.shape)
        #mysavedataset.save_feature_77G(h,x_train,y_train)
        for i in range (len(x_train)):
            datax.append(x_train[i])
            datay.append(y_train[i])
    datax=np.array(datax)
    datay=np.array(datay)
    return datax,datay

def quick_63G_dataset():
    for h in range(1,130):
        print(h)
        x_train0,y_train=mysavedataset.read_dataset_63G(h)
        print("HSAPE:",x_train0.shape,y_train.shape)
         
        n_axis=x_train0.shape[2]
        #x_train0=mynormlization.normlization(x_train0)
        x_train_o=x_train0
        y_train_o=y_train
        C,CC=myfeature1.get_tsfresh(x_train0,y_train)
        C=C.reshape(-1,n_axis*CC)
        E=myfeature2.get_ffeature(x_train0)
        x_train=np.concatenate((C,E), axis=1)
        #x_train=mynormlization.normlization2(x_train)   
        y_train=np.array(y_train)
        print("SHAPE:",x_train.shape,y_train.shape)
        mysavedataset.save_feature_63G(h,x_train,y_train)

if __name__ == '__main__':
    #init_63G_dataset()
    #init_own_dataset()
    init_77G_dataset()