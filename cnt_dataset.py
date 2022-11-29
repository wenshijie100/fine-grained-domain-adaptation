# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:55:02 2021

@author: wensh
"""
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import new_save_dataset as mysavedataset


def file_cnt(file_name,dataset_name):
    path='../'+file_name+'/'+dataset_name
    for root ,dirs ,files in os.walk(path):
        return int((len(files)/3)-1)

def cnt_dataset(dataset_name,percent=1.0):
    dn=file_cnt("dataset_weight",dataset_name)
    seq=[i+1 for i in range(dn)]
    cnt=[0 for i in range(5)]
    
    for i in range(dn):
        datax,datay,dataw=mysavedataset.read_dataset_W(dataset_name,seq[i])
        cnt[int(datay[0])]+=len(datay)
    print(cnt)

if __name__ == "__main__":
    # run the experiment
    cnt_dataset("WISDM")
    #plot_data()
