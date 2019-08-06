import numpy as np

a = np.arange(25).reshape(5, 5)
print(a)

blue = a[[0, 1, 2, 3], [1, 2, 3, 4]]
print("blue is \n", str(blue))

divisby3 = a % 3 == 0
print("divisible by 3 \n", str(a[divisby3]))

output = np.empty_like(a, dtype='float')
print(output.fill(np.nan))

output[divisby3] = a[divisby3]

print(output)
<<<<<<< HEAD:Code snippets/np_array_practice.py
print("test")
=======
testing

>>>>>>> 2d157e3c4c58172ebe7d35858be72a2dbea9aa23:Hello world.py
