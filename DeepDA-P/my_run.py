import os
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
import my_save_dataset as mysavedataset
import my_init
import xgboost as xgb
dn=[75,422,130]
def make_own_train_test(percent=0.0):
    #print(123)
    K=int(percent*dn[0]+0.5)
    X_train,Y_train=[],[]
    X_test,Y_test,L_test=[],[],[]
    
    seq=[i+1 for i in range(dn[0])]
    random.shuffle(seq)
    for i in range(dn[0]):
        #datax,datay=mysavedataset.read_feature_own(seq[i])
        datax,datay=my_init.quick_own_dataset_one_people(seq[i])
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
        datax,datay=mysavedataset.read_feature_77G(seq[i])
        #datax,datay=my_init.quick_77G_dataset_one_people(seq[i])
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
def make_63G_train_test(percent=0.0):
    K=int(percent*dn[2]+0.5)
    X_train,Y_train=[],[]
    X_test,Y_test,L_test=[],[],[]
    
    seq=[i+1 for i in range(dn[2])]
    random.shuffle(seq)
    for i in range(dn[2]):
        datax,datay=mysavedataset.read_feature_63G(seq[i])
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
def run(x_train,y_train,x_test,y_test,L_test):
    '''
    x_combine=x_train
    length=len(x_train)
    for i in range(len(x_test)):
        x_combine.append(x_test[i])
    x_combine=mynormlization.normlization2(x_combine)
    x_train=x_combine[:length]
    x_test=x_combine[length:]
    '''
    print("SSS:",len(x_train),len(y_train),len(x_test),len(y_test))
    #print(L_test)
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    print("svm",clf.score(x_train, y_train),clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)
    #print(y_pred)
    #y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
   
    et = ExtraTreesClassifier(n_estimators=300, max_depth=None,min_samples_split=3, random_state=0)
    et.fit(x_train, y_train)
    print("et",et.score(x_train, y_train),et.score(x_test, y_test))
    y_pred = et.predict(x_test)
    #print(y_pred)
    #y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,min_samples_split=3, random_state=0,bootstrap=True)
    rf.fit(x_train, y_train)
    print("rf",rf.score(x_train, y_train),rf.score(x_test, y_test))
    y_pred = rf.predict(x_test)
    #print(y_pred)
    #y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
    
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    dt.fit(x_train, y_train)
    print("dt",dt.score(x_train, y_train),dt.score(x_test, y_test))
    y_pred = dt.predict(x_test)
    #print(y_pred)
    #y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
    
    nb = GaussianNB().fit(x_train, y_train)
    nb.fit(x_train, y_train)
    print('nb',nb.score(x_train, y_train),nb.score(x_test, y_test))
    y_pred = nb.predict(x_test)
    #print(y_pred)
    #y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
    
    
   
    gbdt=GradientBoostingClassifier(n_estimators=100)
    gbdt.fit(x_train, y_train)
    print("gbdt",gbdt.score(x_train, y_train),gbdt.score(x_test, y_test))
    
    y_pred = gbdt.predict(x_test)
    #print(y_pred)
    #y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
    y_pred_train=gbdt.predict(x_train)
    C=confusion_matrix(y_train, y_pred_train)
    print(C)
    
    #check_data(x_train)
    #check_data(x_test)
    
    '''
    pca = PCA(n_components=100) #降维
    print("HSAPE:",x_train.shape,y_train.shape,x_test.shape,y_test.shape)    
    pca=pca.fit(x_train) #模型训练
    x_train=pca.fit_transform(x_train)
    x_test=pca.fit_transform(x_test)
    '''
    print("HSAPE:",x_train.shape,y_train.shape,x_test.shape,y_test.shape,L_test)
    x_train=mynormlization.check_data(x_train)
    x_test=mynormlization.check_data(x_test)
    rf = RandomForestClassifier(n_estimators=500, criterion='entropy',max_depth=100,min_samples_split=5, random_state=0,bootstrap=True)
    
    #rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10,min_samples_split=2, random_state=0,bootstrap=True)
    rf.fit(x_train, y_train)
    acc_train_single,acc_test_single=rf.score(x_train, y_train),rf.score(x_test, y_test)
    print("rf2",acc_train_single,acc_test_single)
    y_pred = rf.predict(x_test)
    last_num=0
    acc_test_human=0
    for i in range(len(L_test)):
        #print("num",i)
        y_pred_human=y_pred[last_num:last_num+L_test[i]]
        y_pred_human=np.array(y_pred_human)
        y_f=[]
        y_a=int(np.mean(y_pred_human)+0.5)
        for j in range(len(y_pred_human)):
            y_f.append(int(y_pred_human[j]))
            #y_pred_human[j]=int(y_pred_human[j])
        #print("DD",y_pred_human)
        #print("DD",np.bincount(y_pred_human))
        #print(y_f)
        y_label_human=np.argmax(np.bincount(y_f))
        y_label_human=y_a
        if y_label_human==y_test[last_num]:
            acc_test_human+=1
        else:
            print("ERROR",y_test[last_num],y_label_human)
        last_num+=L_test[i]
    acc_test_human=float(acc_test_human/len(L_test))
    #print(y_pred)
    y_pred=(y_pred+0.5).astype(int)
    #print(y_pred)
    C=confusion_matrix(y_test, y_pred)
    print(C)
    print("rf",acc_train_single,acc_test_single,acc_test_human)
    return acc_train_single,acc_test_single,acc_test_human
epoch=10
A,B,C=0.0,0.0,0.0
from sklearn.manifold import TSNE
from sklearn import manifold, datasets

def tsne_change(iris):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(iris)
    print("S2:",X_train.shape,X_tsne.shape)
    
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min) 
    X_norm=np.array(X_norm)
    return X_norm

    
simpel_test=0
for i in range(epoch):
    print("IIIII:",i)
    #X_train,Y_train=make_own_train_test(0)
    X_train,Y_train,X_test,Y_test,L_test=make_77G_train_test(0.3)
    if simpel_test==1:
        X_train=X_train[::2]
        Y_train=Y_train[::2]
        #X_test=X_train[::2]
        #Y_test=Y_train[::2]
    
    #X_train=np.concatenate((X_train,X_train2),axis=0)
    #Y_train=np.concatenate((Y_train,Y_train2),axis=0)
    Y_train=np.array(Y_train)
    Y_train=Y_train.astype(int)
    Y_test=np.array(Y_test)
    Y_test=Y_test.astype(int)
    X_data=np.concatenate((X_train,X_test),axis=0)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    X_data=np.array(X_data)
    '''
    X_data=X_data[:,0:10]
    print("ASHAPE:",X_train.shape,X_test.shape,X_data.shape)
    X_data=tsne_change(X_data)
    L_X_train=len(X_train)

    X_train=X_data[:L_X_train]
    X_test=X_data[L_X_train:]
    print("BSHAPE:",X_train.shape,X_test.shape)
    '''
    #X_train,Y_train,X_test,Y_test,L_test=make_77G_train_test(0.2)
    acc_train_single,acc_test_single,acc_test_human=run(X_train,Y_train,X_test,Y_test,L_test)
    A=A+acc_train_single
    B=B+acc_test_single
    C=C+acc_test_human
print(A/epoch,B/epoch,C/epoch)
    