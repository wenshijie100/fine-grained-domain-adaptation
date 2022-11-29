# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:23:17 2021

@author: wensh
"""

import matplotlib.pyplot as plt 
import numpy as np
import numpy.fft as fft

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示符号

def get_signal(Fs,L,S,sum_len=8):
    T = 1/Fs;             # 采样周期
    t = [i*T for i in range(L)]
    t = np.array(t)
    x = np.linspace(0, 100, 1000)
    plt.show()
    #S = 0.2+0.7*np.cos(2*np.pi*50*t+20/180*np.pi) + 0.2*np.cos(2*np.pi*100*t+70/180*np.pi) ;
    
    
    complex_array = fft.fft(S)
    #print(complex_array.shape)  # (1000,) 
    #print(complex_array.dtype)  # complex128 
    #print(complex_array[1])  # (-2.360174309695419e-14+2.3825789764340993e-13j)
    '''
    #################################
    plt.subplot(411)
    plt.grid(linestyle=':')
    plt.plot(1000*t[1:51], S[1:51], label='S')  # y是1000个相加后的正弦序列
    plt.xlabel("t（毫秒）")
    plt.ylabel("S(t)幅值")
    plt.title("叠加信号图")
    plt.legend()
    
    ###################################
    plt.subplot(412)
    S_ifft = fft.ifft(complex_array)
    # S_new是ifft变换后的序列
    plt.plot(1000*t[1:51], S_ifft[1:51], label='S_ifft', color='orangered')
    plt.xlabel("t（毫秒）")
    plt.ylabel("S_ifft(t)幅值")
    plt.title("ifft变换图")
    plt.grid(linestyle=':')
    plt.legend()
    print(S_ifft)
    ###################################
    '''
    # 得到分解波的频率序列  
    freqs = fft.fftfreq(t.size, t[1] - t[0])
    # 复数的模为信号的振幅（能量大小）
    pows = np.abs(complex_array)
    
    '''
    plt.title('FFT变换,频谱图')
    plt.xlabel('Frequency 频率')
    plt.ylabel('Power 功率')
    plt.tick_params(labelsize=10)
    plt.grid(linestyle=':')
    plt.plot(freqs[freqs > 0], pows[freqs > 0], c='orangered', label='Frequency')
    print(pows[freqs > 0])
    print(len(pows[freqs > 0]))
    #print(freqs[freqs > 0])
    #print(len(freqs[freqs > 0]))
    plt.legend()
    plt.tight_layout()
    '''

    #sum_len=8
    plt.show()
    pows=np.append(pows,0)
    t_len=int(len(pows)/sum_len)
    new_data=[]
    for i in range(sum_len):
        s=0
        for j in range(t_len):
            s=s+pows[i*t_len+j]
        new_data.append(s)
    new_data=np.asarray(new_data,dtype = np.float32)
    #print(new_data)
    return new_data