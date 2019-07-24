import numpy as np
import random

# Generate sample tracks
trackOrigins = {}
for num in range(10):
    trackOrigins[num] = np.array([random.randint(0, 1023), random.randint(0, 1023)])
print(trackOrigins)