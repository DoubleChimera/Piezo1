import math
import numpy as np


NAN = np.nan

X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([2, 4, 8, 16, 32])
LAGTIME = 1
MOMENT = 2


MSD = (np.nansum((X1[LAGTIME:] - X1[:-LAGTIME])**MOMENT) / len(X1[LAGTIME:]))
RMSD = math.sqrt(MSD)

print(MSD)
print(RMSD)
print(math.sqrt(np.nansum(X1[:])))
