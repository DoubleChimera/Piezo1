#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# opens and displays a plot with an image as a background and plots a scatter plot on top of it
# need to CONSIDER A PICK EVENT WITH A LIVE MAP TO SHOW ALL THE DATA POINTS AND ALLOW MOUSE EVENTS
# TO HANDLE THEIR APPEARANCE AND DISAPPEARANCE

import matplotlib.pyplot as plt
im = plt.imread('/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/test_images/test1.tiff')
implot = plt.imshow(im)

# put a blue dot at (10, 20)
plt.scatter([10], [20])

# put a red dot, size 40, at 2 locations:
plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)

plt.show()

# Define a list of points in terms of t,x,y, just like the json files
# make sure it contains at least 3 separate tracks with 10 frames each.

# generate a new list corresponding to track numbers in previous list
# element 1 is track 1, ele 2 is track 2, etc.  and have a 'on/off' column
# designated by 0's and 1's

# use a pick event and sentdex's live chart to handle turning on and off tracks
