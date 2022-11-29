# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:40:59 2021

@author: wensh
"""
import numpy as np
from scipy import signal
def get_body(data):
    newdata = np.copy(data)
    b, a = signal.butter(3, 0.95,'lowpass')
    newdata= signal.lfilter(b, a, data)  # data为要过滤的信号
    return newdata
def get_body2(data):
    newdata = np.copy(data)
    b, a = signal.butter(3, 0.05,'highpass')
    newdata= signal.lfilter(b, a, data)  # data为要过滤的信号
    return newdata
def get_grave2(data):
    newdata = np.copy(data)
    b, a = signal.butter(3, 0.05,'lowpass')
    newdata= signal.lfilter(b, a, data)  # data为要过滤的信号
    return newdata
