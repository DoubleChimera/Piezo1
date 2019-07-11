#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:29:42 2017

@author: andrewgeragotelis

Minor modifications on Wed April 24, 2019, 17:13:33
@editor: Vivek Tyagi
"""
import numpy as np
import pandas as pd
import json, codecs
import os.path

# change working directory
#os.chdir('/Users/andrewgeragotelis/Documents/research_imac/piezo_diffusion/particle_tracks')
#save_path = '/home/vivek/Tobias_Group/Single Particle Tracking -Piezo1 Protein/Piezo1_Trajectory/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_5_MMStack_Pos0/'

# set desired min number of frames per track
#minfrm = 20

def gen_indiv_tracks(save_path, minfrm):
    """
    returns lst[] extracted from a .json file saved by flika's pynsight plugin with
    track lengths greater than the minfrm (minimum frame) value.

    The track number is not related to the track numbers in the .json file,
    they are generated while populating the lst[] with tracks greater than the
    minfrm (minimum frame) value.  It is used as an index for reference only.

    lst is a 3D array. Every element is a particle localization. The columns are [t, x, y], where t is the frame the
    particle was localized in. x and y are the coordinates of the center of the particle determined by a 2D gaussian
    fit, in pixel space.

    lst is a list of track arrays. Each value in a lst array contains an index to a particular track.
    To extract the coordinates of the ith track, use:

        >>> print(lst[i-1])

            array([[   0.   ,   23.32 ,  253.143],
                   [   1.   ,   23.738,  252.749],
                   [   2.   ,   23.878,  252.8  ]])

    gen_indiv_tracks takes two arguments, a save_path and minfrm where
    save_path selects a directory to save .to_csv formatted .txt files for each track individually and
    minfrm is a user selected value for the minimum number of frames allowed in each tracks

    This function is designed to be run independently with arguments, or called from another function
    """
    # run file from Kyle to read .json files
    #execfile("open_tracks.py")
    exec(open("/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/open_tracks.py").read())
    numTracks = len(tracks)

    # make list of tracks with >= min number frames
    lst = []
    for i in range(0,(numTracks)):
        track = tracks[i]
        pts = txy_pts[track, :]
        if len(pts) >= minfrm:
            lst.append(pts)

    for k in range(0,len(lst)):
        df = pd.DataFrame(lst[k])
        num = k + 1
        completeName = os.path.join(save_path,'track%i.txt' % num)
        df.to_csv(completeName, index=False, header=['Frame_Number','X-coordinate','Y-coordinate'])
    return lst

if __name__ == '__main__':
    save_path = '/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/Sample Track Files/'
    minfrm = 20
    lst = gen_indiv_tracks(save_path, minfrm)
