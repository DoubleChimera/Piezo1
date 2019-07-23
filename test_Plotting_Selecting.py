# setup a list of arrays with origin points
import random
import numpy as np
import os.path
from fileinput import filename

# Generate sample tracks
trackOrigins = {}
for num in range(10):
    trackOrigins[num] = np.array([random.randint(0, 1023), random.randint(0, 1023)])
print(trackOrigins)

# Set path to .tif file
tifFile = r''

# End of initial info setup
