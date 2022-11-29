import numpy as np

A = np.ones((1, 32, 30, 3))
print(A.shape)
A=A.transpose(0,3,1,2)
print(A.shape)
