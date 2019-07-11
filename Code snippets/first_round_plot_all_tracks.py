# SOME NOTES:
# pictures will be 1024 pixels by 1024 pixels
# measurement of each pixel in microns will be provided per experiment

# this function is supposed to:
# take a single track file and make a plot with
# the same pixel resolution as the movie it came from (1024 x 1024)
# and the same size as the movie frame
# on a clear background for an overlay
# on top of the movie frames
"""
    Or, instead of the above notes, use the lst[] array generated in get_tracks..
    and parse through it to produce a plot for each track.
    put it on a clear background and overlay it on a frame of the movie
    use pixel size to adjust it accordingly.  Look at the mathematica file to
    determine good arguments and return values for the next steps.

    Probably this is the way to go. Just make sure you're also outputting the
    relevant files at important steps for comparison. clear backgrounds to allow
    overlaying would be nice for these output files as well.

    some ideas for input arguments, the lst[] array of data, the movie file, the output directory, frame duration, pixel width, number of pixels,
    some ideas for return values, ... tbd look at mathematica structure
"""
import pandas as pd
import numpy as np
import os.path
import csv

# Step 1: load a single track file into memory
file_path = '/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/Sample Track Files/track1.txt'
# Step 2: Convert file into an array
# The first list in the array is the header, indexed at 0
# The first frame, x, y is indexed as list 1 in the array

results = []
with open(file_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader: # each row is a list
        results.append(row)
print(results)
# Step 3: Plot the data onto a 768 x 768 pixel Plot
#         origin in upper left, maximum bottom right
#         add title, axis labels, major and minor tick marks
