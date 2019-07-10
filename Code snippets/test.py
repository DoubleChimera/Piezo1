import numpy as np
import math

NaN = np.NaN

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 4, 8, 16, 32])
lagtime = 1
moment = 2


msd = (np.nansum((x1[lagtime:] - x1[:-lagtime])**moment) / len(x1[lagtime:]))
rmsd = math.sqrt(msd)

print(msd)
print(rmsd)
print(math.sqrt(np.nansum(x1[:])))
