# setup a list of arrays with origin points
import random
import numpy as np
from fileinput import filename
import matplotlib.pyplot as plt

# Generate sample tracks
trackOrigins = {}
for num in range(10):
    trackOrigins[num] = np.array([random.randint(25, 1000), random.randint(25, 1000)])
print(trackOrigins)
print("----------")

# Set path to .tif file
tifFile = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/test_images/test1.tiff'

# End of initial info setup
# ! Start of actual coding for this issue
# Start code below this --
img = plt.imread(tifFile)
imgplot = plt.imshow(img)


# plt.show()

# * CURRENT DEBUGGING CODE BELOW
# extract all x-y values as a list of arrays named xyOrigins
xyOrigins = []
for xycoords in trackOrigins.values():
    xyOrigins.append(xycoords)

# Using list comprehension to extract x and y vals
xvals = [coord[0] for coord in xyOrigins]
yvals = [coord[1] for coord in xyOrigins]

# plot the xy-values on a scatter plot with a tif background
plt.scatter(x=xvals, y=yvals)
plt.show()
# move the plot to the lasso code so points can be selected

# use the selected points to determine which tracks they are from

# produce a final output of useable tracks for further calcs



# OLD DEBUGGING CODE BELOW