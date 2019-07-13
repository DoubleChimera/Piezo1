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
            Turn on/off tracks by selection
                1. Start by turning off one track in the plot
                2. Now use a list of indices for tracks and turn them off
                3. Now select a point in 1 track and use it to determine the closest track and turn it off
                4. Now select an area on the plot and turn off all tracks within that bounds (selection rectangle)
                5. Link the previous with a mouse-enabled selection box
            Look at matplotlib for handling mouse selection area events to change point colors, add them to a list, etc.
"""


# todo Step 4:  Take in manual inputs for box corners with which to select points, turning them 'on' or 'off'
# todo          for future calculations, dim them on the plot to indicate status
# todo Step 5:  With selected points, calculate RMSD and plot, for all tracks
# todo NOTE     Make sure there are relevant outputs produced at each step into a useful folder name,
# todo          consider toggling these outputs somehow
#endregion
# * testing testing... 1... 2... 3...
