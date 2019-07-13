#region
# todo Step 1:  Load a .json file, a corresponding .tif file, set an output directory,
# todo          and take in initial inputs for the file
# //   Step 2:  Convert the .json file into a numpy array, find and fill any blank spots with a NaN value
# todo Step 3:  Grab the first frame of the .tif file and display, plot all tracks on top of .tif
# todo          (color code by length), and provide relevant stats, such as number of frames and length of trajectory.
# todo          Leave room for future to adjust .tif picture quality (brightness, contrast, etc.)
""" Some thoughts on step 3:
            Get the plot ready
                1. Load a .tif file
                2. Extract the first frame and display
                3. Put axes for pixels on display, 0,0 top left, 1024, 1024 bot right
                4. Be able to plot a track on top of that layer with a list of np.arrays
            Look at matplotlib for handling mouse selection area events to change point colors, add them to a list, etc.
"""


# todo Step 4:  Take in manual inputs for box corners with which to select points, turning them 'on' or 'off'
# todo          for future calculations, dim them on the plot to indicate status
# todo Step 5:  With selected points, calculate RMSD and plot, for all tracks
# todo NOTE     Make sure there are relevant outputs produced at each step into a useful folder name,
# todo          consider toggling these outputs somehow
#endregion
# * testing testing... 1... 2... 3...
