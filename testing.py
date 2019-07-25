import numpy as np
import copy

a = np.array([1, 100, 200])
b = np.array([2, 101, 201])
c = np.array([3, 102, 202])

test = np.array([a, b, c])

test1 = np.array([test, test*2, test*3])

print(test1[0])
bet = []
bet.append(test1[0])
xvals = bet[0][:,1]
yvals = bet[0][:,2]

print(xvals)
print(yvals)