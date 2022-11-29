# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 00:27:36 2021

@author: wensh
"""
import numpy as np
def save_dataset(dataset_name,num,datax,datay):
    path='../dataset_tree/'+dataset_name+'/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 
def save_feature(dataset_name,num,datax,datay):
    path='../dataset_tree/'+dataset_name+'/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay)

def read_dataset(dataset_name,num):
    path='../dataset_tree/'+dataset_name+'/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay

def save_dataset_W(dataset_name,num,datax,datay,dataw):
    path='../dataset_weight/'+dataset_name+'/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay)
    np.save(path+'-W',dataw)

def read_dataset_W(dataset_name,num):
    
    path='../dataset_weight/'+dataset_name+'/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    dataw=np.load(path+'-W.npy')
    return datax,datay,dataw
    
def save_datasetO_original(dataset_name,num,datax,datay):
    path='../dataset_tree/'+dataset_name+'/'+str(num)
    np.save(path+'-X',datax)
    np.save(path+'-Y',datay) 

def read_dataset_original(dataset_name,num):
    path='../dataset_tree/'+dataset_name+'/'+str(num)
    datax=np.load(path+'-X.npy')
    datay=np.load(path+'-Y.npy')
    return datax,datay
