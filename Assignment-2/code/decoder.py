import numpy as np


T = 10*np.ones((10,5,10))
R = np.ones((10,5,10))
V = np.arange(0, 10)

print(np.max(np.sum(T*(R+V), axis=2), axis=1))