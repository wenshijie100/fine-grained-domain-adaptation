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
    #print(len(fname_divide))
    #print(fname_divide)     
    return fname_divide
def check_illegal(xs,ys,zs,cs):
    if mytree.get_fangcha(xs)+mytree.get_fangcha(ys)+mytree.get_fangcha(zs)<0.5 and cs!=0:
        return 0
    if np.max(xs)>25 or np.max(ys)>25 or np.max(zs)>25:
        cnt_x_25,cnt_y_25,cnt_z_25=0,0,0
        for i in range(len(xs)):
            if xs[i]>15:
                cnt_x_25+=1
            if ys[i]>15:
                cnt_y_25+=1
            if zs[i]>15:
                cnt_z_25+=1
        if  cnt_x_25<=3 and  cnt_y_25<=3 and  cnt_z_25<=3:
            return 0
    if np.min(xs)<-25 or np.min(ys)<-25 or np.min(zs)<-25:
        cnt_x_25,cnt_y_25,cnt_z_25=0,0,0
        for i in range(len(xs)):
            if xs[i]<-15:
                cnt_x_25+=1
            if ys[i]<-15:
                cnt_y_25+=1
            if zs[i]<-15:
                cnt_z_25+=1
        if cnt_x_25<=3 and  cnt_y_25<=3 and  cnt_z_25<=3:
            return 0
    
    return 1
def load_dataset(path="Tremor_LR",h=5):
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
        #print("LLLL",line[1])
    #print("TTTTTTTTTTTTTTTTTTT",d['5211158e62cb7effea3f6f16d6a5bd8a.txt'])
   
    N_TIME_STEPS = 128
    N_FEATURES = 3
    step = 128
  

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
        '''
        if index!=-1:
            fname=filename[:index]+'.txt'
        '''
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
        
        #重力波
        X1=mygetbody.get_body2(X)
        Y1=mygetbody.get_body2(Y)
        Z1=mygetbody.get_body2(Z)    
        
        X2=mygetbody.get_grave2(X)
        Y2=mygetbody.get_grave2(Y)
        Z2=mygetbody.get_grave2(Z)
        '''
        if k==1:
                
            plt.figure(100)
            plt.plot(X)
            plt.plot(Y)
            plt.plot(Z)
            plt.show()
        '''  
        #if mytree.get_fangcha(X)+mytree.get_fangcha(Y)+mytree.get_fangcha(Z)<1 and d[fname]!=0:
            
            #continue
        pre_label=[]
        for i in range(0, L- N_TIME_STEPS, step):
            if i<100:
                continue
            xs = X[i:i+N_TIME_STEPS]
            ys = Y[i:i+N_TIME_STEPS]
            zs = Z[i:i+N_TIME_STEPS]
            
            xs1 = X1[i:i+N_TIME_STEPS]
            ys1 = Y1[i:i+N_TIME_STEPS]
            zs1 = Z1[i:i+N_TIME_STEPS]
            
            xs2 = X2[i:i+N_TIME_STEPS]
            ys2 = Y2[i:i+N_TIME_STEPS]
            zs2 = Z2[i:i+N_TIME_STEPS]
            if check_illegal(xs,ys,zs,d[fname])==0:
                pre_label.append(0)
                continue
            else:
                pre_label.append(1)
            m=[xs,ys,zs]
            m=np.asarray(m, dtype= np.float32)
            m=m.T
           
            segments.append(m)
            #print("i",i,filename)
            labels.append(d[fname])
        C=C+1
        #segments,labels=mytree.K_tree(segments,labels,pre_label,h)
        #print("OMG",len(segments),L/128)
        if len(segments)>=2:
            cnt=cnt+1
            mysavedataset.save_dataset_own(cnt,segments,labels)
            #print(len(segments),len(labels))
        else :
            continue
        for i in range(len(segments)):
            X_train.append(segments[i])
            Y_train.append(labels[i])   
        segments,labels= mysavedataset.read_dataset_own(cnt)
        
   # X_train=normlization3(X_train)
   # X_train=check_data(X_train)
    X_train=np.asarray(X_train,dtype = np.float32)
    #X_test=np.asarray(X_test,dtype = np.float32)
    Y_train=np.asarray(Y_train,dtype = np.float32)
    #Y_test=np.asarray(Y_test,dtype = np.float32)
    print("check_shape_own:",X_train.shape, Y_train.shape, Y_train.shape,np.max(Y_train))
    return X_train,  Y_train 
if __name__ == '__main__':
    load_dataset()