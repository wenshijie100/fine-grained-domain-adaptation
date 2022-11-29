# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 00:27:36 2021

@author: wensh
"""
import numpy as np
def save_dataset_77G(num,datax,datay):
    path='../dataset_tree/77G/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 
def save_feature_77G(num,datax,datay):
    path='../feature_tree/77G/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay)

def read_dataset_77G(num):
    path='../dataset_tree/77G/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay

def read_feature_77G(num):
    path='../feature_tree/77G/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay
    
def save_datasetO_77G(num,datax,datay):
    path='../dataset_O/77G/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 

def read_datasetO_77G(num):
    path='../dataset_O/77G/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay

##########################################own 
def save_dataset_own(num,datax,datay):
    path='../dataset_tree/own/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 
def save_feature_own(num,datax,datay):
    path='../feature_tree/own/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 
    

def read_dataset_own(num):
    path='../dataset_tree/own/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay
def read_feature_own(num):
    path='../feature_tree/own/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay
###########################################63    
def save_dataset_63G(num,datax,datay):
    path='../dataset_tree/63G/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 
def save_feature_63G(num,datax,datay):
    path='../feature_tree/63G/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay)

def read_dataset_63G(num):
    path='../dataset_tree/63G/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay

def read_feature_63G(num):
    path='../feature_tree/63G/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay