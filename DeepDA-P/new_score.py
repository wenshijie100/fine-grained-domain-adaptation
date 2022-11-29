# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:30:17 2022

@author: wensh
"""
import numpy as np
import math
import my_gauss as mygauss
import new_feature as myfeature
import my_tree as mytree
#check noise window
def check_illegal(xs,ys,zs,cs):
    if mytree.get_fangcha(xs)+mytree.get_fangcha(ys)+mytree.get_fangcha(zs)<1 and cs!=0:
        return 0
    allow_range=[-15,15,-10,10]
    if np.max(xs)> allow_range[1] or np.max(ys)>allow_range[1] or np.max(zs)>allow_range[1]:
        cnt_x_25,cnt_y_25,cnt_z_25=0,0,0
        for i in range(len(xs)):
            if xs[i]>allow_range[3]:
                cnt_x_25+=1
            if ys[i]>allow_range[3]:
                cnt_y_25+=1
            if zs[i]>allow_range[3]:
                cnt_z_25+=1
        if  cnt_x_25<=3 and  cnt_y_25<=3 and  cnt_z_25<=3:
            return 0
    if np.min(xs)<allow_range[0] or np.min(ys)<allow_range[0] or np.min(zs)<allow_range[0]:
        cnt_x_25,cnt_y_25,cnt_z_25=0,0,0
        for i in range(len(xs)):
            if xs[i]<allow_range[2]:
                cnt_x_25+=1
            if ys[i]<allow_range[2]:
                cnt_y_25+=1
            if zs[i]<allow_range[2]:
                cnt_z_25+=1
        if cnt_x_25<=3 and  cnt_y_25<=3 and  cnt_z_25<=3:
            return 0
    return 1


#cnt similarity(cos)
def norm(vector):
    return math.sqrt(sum(x * x for x in vector))    
 
def cosine_similarity(vec_a, vec_b):
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    #print("COS:",-dot / (norm_a * norm_b))
    return dot / (norm_a * norm_b)

def get_score(X_train,Y_train,K=6):
    #print("Y:",Y_train)
    last_num=0
    tree=[]
    last_i=0
    K=min(K,len(X_train))
    print(len(X_train))
    i=0
    while(i+K<=len(X_train)):
        t=[]
        for j in range(K):
            t.append(list(X_train[i+j]))
        tree.append(t)
        i=i+K
        last_i=i
    #print("D",last_i,len(X_train))
    while(last_i<len(X_train)):
        tree[-1].append(X_train[last_i])
        last_i+=1
    
    score=[]
    f=[]
    tree=np.array(tree)
    original_K=K
    for i in range(len(tree)):
        K=len(tree[i])
        #print("TREE",tree[i][0])
        for j in range(K):
            f.append(myfeature.get_ffeature(tree[i][j]))    
        mat=np.zeros((K,K))
        for j in range(K):
            mat[j][j]=1
            for k in range(j+1,K):
                mat[k][j]=cosine_similarity(f[i*original_K+j],f[i*original_K+k])
                mat[j][k]=mat[k][j]
        smat=np.sum(mat,axis=1)  
        
        '''
        max_number=np.argmin(smat)
        bmat=np.zeros(K)
        bmat[max_number]=-smat[max_number]
        for j in range(K):
            if j!=max_number:
                mat[max_number][j]=0
        print("MAT:",mat)
        print("bmat",bmat)
        '''
        tmp_score=smat/K
        #print(tmp_score)
        cnt=0
        for j in range(K):
            tree[i][j]=np.array(tree[i][j])
            if check_illegal(tree[i][j][:,0],tree[i][j][:,1],tree[i][j][:,2],Y_train[j])==0:
                tmp_score[j]=0.01
                cnt=cnt+1
            score.append(tmp_score[j])
    score=np.array(score)
   
    if(np.mean(score)<0.5):
        print("CNT_illegal",cnt,len(score),Y_train[0],np.mean(score))
        return score*0.1,0
    return score*np.mean(score),1

        
        