# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:49:22 2021

@author: wensh
"""




from sklearn.cluster import DBSCAN
import numpy as np
import math
import pandas as pd
import tsfresh as tsf
import my_normlization as mynormlization
import matplotlib.pyplot as plt
color=['b','orange','g','y','c']
def get_fangcha(datax):
    datax=np.array(datax)
    sum_t=np.var(datax)
    #print("S:",sum_t,datax.shape[0],datax)
    return float(sum_t)
def plot_combine(datax,datay,y_mode=1,num=0):
    X_train_o=[]

    plt.figure(num)
    L=datax.shape[1]
    lx = np.linspace(0,L*datax.shape[0],L*datax.shape[0]+1)
    #print(datax.shape,lx)
    for k in range(datax.shape[0]):
        for i in range(datax.shape[2]):
            X=datax[k,:,i]
            t=lx[k*L:(k+1)*L]
            #print("LL",len(t),len(X))
            if datay[k]==y_mode:
                plt.plot(lx[k*L:(k+1)*L],X,color[i])
                if i==0:
                    X_train_o.append(datax[k])
            else :
                plt.plot(lx[k*L:(k+1)*L],X,'k')
        #plt.xlim(0, 8)
    plt.show()
    X_train_o=np.array(X_train_o)
    return X_train_o
def check_noise(datax,datay,y_mode=1,num=0):
    X_train_o=[]

    L=datax.shape[1]
    lx = np.linspace(0,L*datax.shape[0],L*datax.shape[0]+1)
    #print(datax.shape,lx)
    for k in range(datax.shape[0]):
        for i in range(datax.shape[2]):
            X=datax[k,:,i]
            t=lx[k*L:(k+1)*L]
            #print("LL",len(t),len(X))
            if datay[k]==y_mode:
                if i==0:
                    X_train_o.append(datax[k])
            
    X_train_o=np.array(X_train_o)
    return X_train_o


def get_energy3(data):
    t=[]
    ts = pd.Series(data)  #数据x假设已经获取
    
    ae = tsf.feature_extraction.feature_calculators.abs_energy(ts)
    t.append(ae)
    
    ae=tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.count_above_mean(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.count_below_mean(ts)
    t.append(ae)
    
    ae = tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.last_location_of_maximum(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.last_location_of_minimum(ts)
    t.append(ae)
    
    '''
    ae=tsf.feature_extraction.feature_calculators.longest_strike_above_mean(ts)
    t.append(ae)
    ae=tsf.feature_extraction.feature_calculators.longest_strike_below_mean(ts)
    t.append(ae)
    '''
    #print(t)
    t=np.array(t)
    return t    


def my_var(datax,avl):
    sum_var=0 
    for i in range (datax.shape[0]):
          sum_var=sum_var+pow(datax[i]-avl,2)
    t=[]
    t.append(sum_var)
    t.append(np.max(datax)-np.min(datax))
    t=np.array(t)
    
    #print("T",t,t.shape)
    return t

def get_tsfresh3(datax):
    new_data=[]
    t_data=[]
    CC2=10
    for i in range(datax.shape[0]):
        t=[1]
        t=np.array(t)
        for j in range(datax.shape[2]):# 30 channels
       
            X=datax[i,:,j]
           
            X=np.array(X)
            X=X-np.mean(X)
            #print("check_X",X)
            A=get_energy3(X)
            B=my_var(X,0)
            #print("T:",t.shape)
            t=np.concatenate((t,A,B,[i]), axis=0)
            #A=1 B=16
            
        t=t[1:]
        t=t.reshape(1,datax.shape[2]*CC2)
        t=np.array(t)
        #print("T:",t.shape)
        #print("T:",t_data.shape)
        if i==0:
             t_data=t
        else :
            t_data=np.concatenate((t_data,t), axis=0)
       
    new_data=t_data
    new_data=np.array(new_data)
    #print(new_data.shape)
    return new_data,CC2 

def check_simliar(datax,datay):
    sum_d=0
    for i in range(len(datax)):
        sum_d=sum_d+pow(datax[i]-datay[i],2)
    return sum_d
def check_cor(A,B):
    t=0
    for i in range(len(A)):
       t=t+abs(A[i]-B[i]) 
    return t

def init_link(K):
    F=[i for i in range(K+1)]
    S=[1 for i in range(K+1)]
    return F,S
def find_link(num,F):
    #print("find_link",num)
    if F[num]==num:
        return F[num]
    else :
        F[num]=find_link(F[num],F)
        return F[num]

class Myclass(object):
    def __init__(self, pos,value,vn,cnt,size):
        self.pos = pos
        self.value = np.array(value)
        self.vn=vn
        self.cnt = cnt
        self.size=size #cnt before delete

    def clean(self):
        self.color = 0
        self.value = np.array([0.0 for i in range(self.vn)])
        self.cnt = 0
tree=[]
st=[]
def insert_tree(num,level,left,right,pos,K,L,data):
    #print("IS:",num,level,left,right,pos)
    global tree
    
    tree[num].cnt+=1
    if left==right:
        tree[num].value=data
        tree[num].pos=pos
        return
    new_left,new_right=left,left+L[level+1]-1
    for new_num in range((num-1)*K+2,+num*K+2):
        
        if new_left<=pos and pos<=new_right:
            insert_tree(new_num,level+1,new_left,new_right,pos,K,L,data)
        new_left+=L[level+1]
        new_right+=L[level+1]
        
def build_tree(vn,cnt,K):#特征维度，总节点数，K叉树
    global tree
    global st
    st=[]
    log_cnt=math.log(cnt,K)
    #print("LOG:",log_cnt,K,cnt)
    if log_cnt==int(log_cnt):
        log_cnt=int(log_cnt+1)
    else:
        log_cnt=int(log_cnt+2)
    L=[pow(K,i) for i in range(log_cnt)]
    L=L[::-1]
    
    t=[0.0 for i in range(vn)]
    tree=[ Myclass(0,t,vn,0)for i in range(pow(K,log_cnt))]
    
   # print(L,pow(K,log_cnt))
    return L

    
def update_tree(num,level,left,right,K,L):
    global tree
    if left==right:
        return 
    tree[num].clean()
    
    new_left,new_right=left,left+L[level+1]-1
    for new_num in range((num-1)*K+2,num*K+2):
        update_tree(new_num,level+1,new_left,new_right,K,L)
        tree[num].cnt+=tree[new_num].cnt
        tmp=tree[new_num].cnt*tree[new_num].value
        #print(tree[num].value)
        #print(tmp)
        tree[num].value+=tmp
        new_left+=L[level+1]
        new_right+=L[level+1]
    if  tree[num].cnt ==0 :
        return
    tree[num].value=tree[num].value/tree[num].cnt 
    tmp_cor=[0.0 for i in range(K)]
    while(True):
        mx=0
        for i in range(K):
            new_i=i+(num-1)*K+2
            if tree[new_i].cnt==0:
                continue
            tmp_cor[i]=check_cor(tree[new_i].value,tree[num].value)
            mx=max(mx,tmp_cor[i])
       
        if mx<0.125*tree[num].vn*(1+float(10-level)/10):
           break
        
        tree[num].clean()
        for i in range(K):
            new_i=i+(num-1)*K+2
            if tree[new_i].cnt==0:
                continue
            if mx==tmp_cor[i]:
                tree[new_i].cnt=0
            tree[num].cnt+=tree[new_i].cnt
            tree[num].value+=tree[new_i].cnt*tree[new_i].value   
        tree[num].value=tree[num].value/tree[num].cnt
    if tree[num].cnt<0.25/(1+float(10-level)/10)*L[level]:
        tree[num].cnt=0
    '''
    F,S=init_link(K)
    for i in range(K):
        new_i=i+(num-1)*K+2
        S[i]=tree[new_i].cnt
    for i in range(K):
        new_i=i+(num-1)*K+2
        if tree[new_i].cnt==0:
            continue
        for j in range(i+1,K):
            new_j=j+(num-1)*K+2
            if tree[new_j].cnt==0:
                continue
            print("COR/VALUE:",num,level,left,right,check_cor(tree[new_i].value,tree[new_j].value)/tree[num].vn)
            if  check_cor(tree[new_i].value,tree[new_j].value)<0.15*tree[num].vn*(1+float(10-level)/20):
                root_i=find_link(i, F)
                root_j=find_link(j, F)
                if root_i==root_j:
                    continue
                if root_i < root_j:
                    F[root_j]=root_i
                    S[root_i]+=S[root_j]
                    S[root_j]=0
            else:
                continue
                print("SSS",num,tree[new_i].value,tree[new_j].value)
    root_num=np.argmax(S)
    if S[root_num]<0.35*tree[num].cnt or S[root_num]<0.25*L[level]:
        print("ERORR",num,level,left,right,S[root_num],tree[num].cnt)
        tree[num].cnt=0
    else :
        tree[num].cnt=S[root_num]
        for i in range(K):
            new_i=i+(num-1)*K+2
            if F[i]==root_num:
                tree[num].value=tree[num].value*1.0+tree[new_i].value*float(tree[new_i].cnt/tree[num].cnt)
    
    '''        
    #print("N:",num,level,left,right,tree[num].cnt)
    #print("N:",tree[num].value[:4])
    return 

def dfs(num,level,left,right,K,L):
    global tree
    global st
    if tree[num].cnt==0:
        return 
    if left==right:
        st.append(tree[num].pos)
        return
    #print("L:",left,right,L[level])
    new_left,new_right=left,left+L[level+1]-1
    for new_num in range((num-1)*K+2,+num*K+2):
        dfs(new_num,level+1,new_left,new_right,K,L)
        new_left+=L[level+1]
        new_right+=L[level+1]

def K_tree(X_train_o,Y_train,h=1,K=6):  
   
    X_train_o=np.array(X_train_o)     
    X_train,CC2=get_tsfresh3(X_train_o)
    X_train=np.array(X_train)
      
    L=build_tree(CC2*3,len(X_train),K)
    '''    
    t=[[1.5,1.5,1.5],[3,3,3],[1,2,2],[3,4,4],
       [2,2,2],[2,1,1],[4,1,1],[2,0,2],
       [1,3,3],[0,3,3],[2,1,1],[1,1,0],
       [2,2,2],[1,0,1],[1,1,3],[0,0,2]]
    t=np.array(t)
    t=mynormlization.normlization3(t)
    
    
    for i in range(len(t)):
        insert_tree(1,0,1,1+L[0]-1,i+1,K,L,t[i])
    '''
    X_train=mynormlization.normlization3(X_train)
    X_train=np.array(X_train)
    #for i in range(len(X_train)):
        #print("XX",X_train[i,4:8])
    for i in range(len(X_train)):
        insert_tree(1,0,1,1+L[0]-1,i+1,K,L,X_train[i])
    update_tree(1,0,1,1+L[0]-1,K,L)
    dfs(1,0,1,1+L[0]-1,K,L)
    '''
    for i in range(len(st)):
        print(X_train[st[i]-1])
    '''
    #print("S:",len(st),st)
   # print(tree[1].cnt,tree[1].value)
    datay=[0 for i in range(len(X_train))]
    for i in range(len(st)):
        datay[st[i]-1]=1
    #X_train_o=plot_combine(X_train_o,datay,1,h)
    X_train_o=check_noise(X_train_o,datay,1,h)

    Y_train=Y_train[:len(X_train)]
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    #print("GGG:",X_train_o.shape,Y_train.shape)
    return X_train_o,Y_train