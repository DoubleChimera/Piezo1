#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np

testarray = [['frame', 'x_val', 'y_val'],[1, 2, 3], [2, 4, 9], [3, 9, 27], [4, 16, 81], [5, 25, 243]]
print(testarray)

# Inefficient transpose due to multiple data type conversions
testarraytranspose = np.array(testarray).transpose().tolist()

# Efficient transpose
#solution1 = map(list, zip(*testarray))
#for x in solution1:
#   print(x[2])

# #the first x coordinate
# print(testarray[1][1])
#
# #the first y coordinates
# print(testarray[1][2])
#
# #the second x coordinate
# print(testarray[2][1])
#
# #the second y coordinate
# print(testarray[2][2])

t_vals = testarraytranspose[0][1:]
x_vals = testarraytranspose[1][1:]
y_vals = testarraytranspose[2][1:]
print(t_vals)
print(x_vals)
print(y_vals)

plt.plot(x_vals, y_vals)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()
