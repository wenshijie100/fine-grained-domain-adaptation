# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:43:01 2021

@author: wensh
"""
def get_F1(confusion_matrix):
    len_labels=len(confusion_matrix)
    accu = [0 for i in range(len_labels)]
    column = [0 for i in range(len_labels)]
    line = [0 for i in range(len_labels)]
    accuracy = 0
    recall = 0
    precision = 0
    for i in range(0,len_labels):
        accu[i] = confusion_matrix[i][i]
    for i in range(0,len_labels):
       for j in range(0,len_labels):
           column[i]+=confusion_matrix[j][i]
    for i in range(0,len_labels):
       for j in range(0,len_labels):
           line[i]+=confusion_matrix[i][j]
    for i in range(0,len_labels):
        accuracy += float(accu[i])/len_labels
    for i in range(0,len_labels):
        if column[i] != 0:
            recall+=float(accu[i])/column[i]
    recall = recall / len_labels
    for i in range(0,len_labels):
        if line[i] != 0:
            precision+=float(accu[i])/line[i]
    precision = precision / len_labels
    f1_score = (2 * (precision * recall)) / (precision + recall)
    print("F1:",f1_score)