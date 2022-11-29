# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:45:54 2021

@author: wensh
"""
import os
import xlrd
import json
import numpy as np
import my_get_body as mygetbody
import my_plot as myplot
import my_tree as mytree
import my_save_dataset as mysavedataset
import new_window as mywindow
def getFlist(file_dir):
    fname_divide=[]
    ct=0
    for root, dirs, files in os.walk(file_dir):
        #print('root_dir:', root)
        #print('sub_dirs:', dirs)
        #print('files:', files)
        for j in range(len(files)):
            ct=ct+1
            ri = files[j].rindex('.')
            fname_divide.append(files[j][:ri])            
    print("CT:",ct)         
    return fname_divide

def load_dataset(path="Tremor_LR",h=5,N_TIME_STEPS = 128,step = 128):
    #读取label
    columns = ['name', 'score']
    #df = read_csv('lable.xlsx', names = columns,header=0)
    fpath='../owndataset/'
    xlsx = xlrd.open_workbook(fpath+'lable.xlsx')
    # 查看所有sheet列表
    #print('All sheets: %s' % xlsx.sheet_names())
    df=xlsx
    sheet1 = xlsx.sheets()[0]
    #print(sheet1.ncols,sheet1.nrows)
    d={}
    d2={}
    for i in range(sheet1.nrows):
        line = sheet1.row_values(i)
        d[line[0]]=line[1]
    filenames = os.listdir(fpath+path)    
    filesum=len(filenames)
    print("Flen",filesum)
    mylist = [i for i in range(filesum)]
    C=0  
    cnt=0
    X_train,Y_train=[],[]
    pname=getFlist(fpath+'divide_'+path)
    for i in range(len(pname)):
        d2[pname[i]]=1
    
    for filename in filenames:
        X,Y,Z=[],[],[]
        #print(filename)
        index=filename.rfind('-')
        fname=filename
    
    for k in range(filesum):
    
        segments = []
        labels= []
        filename=filenames[k]
        X,Y,Z=[],[],[]
        #print(filename)
        index=filename.rfind('-')
        fname=filename
        #print(fname)
        if d.get(fname,-1)==-1 or d2.get(fname,-1)==-1:
            #print("NOT FIND",fname)
            continue
        '''
            t = time.time()
            t=int(round(t * 1000000)) 
            if t%5!=0:
                continue
        '''
        f = open(fpath+path+'/'+filename,"r",encoding="utf-8")
        p = json.load(f)
        L=len(p['data']['acc'])
        for i in range(L):
            X.append(p['data']['acc'][i]['x'])
            Y.append(p['data']['acc'][i]['y'])
            Z.append(p['data']['acc'][i]['z'])
        cnt,segments,labels=mywindow.slide_window('own',cnt,X,Y,Z,d[fname],N_TIME_STEPS,step)
        if len(segments)==0:
            continue
        for i in range(len(segments)):
            X_train.append(segments[i])
            Y_train.append(labels[i])   
 
    X_train=np.asarray(X_train,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    print("check_shape_own:",X_train.shape, Y_train.shape, Y_train.shape)
    mywindow.cnt_class(Y_train)
    return X_train,  Y_train 
if __name__ == '__main__':
    load_dataset()