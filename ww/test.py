import numpy as np

a = np.zeros((2,3))

def fun(f):
    f[0][1] = 1

fun(a)
print(a)
