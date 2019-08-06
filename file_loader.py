# -*- coding: utf-8 -*-
import codecs
import json
import os.path
from fileinput import filename

import numpy as np
import pandas as pd


def open_tracks(filename):
    """
    returns txy_pts and tracks extracted from a .json file saved by flika's pynsight plugin

    txy_pts is a 2D array. Every row is a particle localization. The columns are [t, x, y], where t is the frame the
    particle was localized in. x and y are the coordinates of the center of the particle determined by a 2D gaussian
    fit, in pixel space.

    tracks is a list of track arrays. Each value in a track array contains an index to a point in txy_pts.
    To extract the coordinates of the ith track, use:

        >>> track = tracks[i]
        >>> pts = txy_pts[track, :]
        >>> print(pts)

            array([[   0.   ,   23.32 ,  253.143],
                   [   1.   ,   23.738,  252.749],
                   [   2.   ,   23.878,  252.8  ]])
    """

    obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
    pts = json.loads(obj_text)
    txy_pts = np.array(pts['txy_pts'])
    tracks = [np.array(track) for track in pts['tracks']]
    return txy_pts, tracks


def gen_indiv_tracks(save_path, minfrm, tracks, txy_pts):
    """
    returns lst[] and lstnan[] and trackOrigins (dictionary)
    lst[] is extracted from a .json file saved by flika's pynsight plugin with
    track lengths greater than the minfrm (minimum frame) value.
    lstnan[] has blank tracks filled with nan values

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

    This function is designed to be run independently with arguments, or be called from another function
    """

    numTracks = len(tracks)
    nan = np.nan

    # make list of tracks with >= min number frames
    lst = []

    for i in range(0,(numTracks)):
        track = tracks[i]
        pts = txy_pts[track, :]
        if len(pts) >= minfrm:
            lst.append(pts)

    # Move all tracks such that their starting index is 0
    for track in range(len(lst)):
        if lst[track][0][0] != 0:
            indset = lst[track][0][0]
            for pts in range(len(lst[track][0:])):
                lst[track][pts][0] = lst[track][pts][0] - indset

    lstnan = np.copy(lst)

    # make a new directory to save all the track files in if it doesn't already exist
    allTracksDir = os.path.join(save_path, 'All_tracks')
    if not os.path.exists(allTracksDir):
        os.makedirs(allTracksDir)

    # parse through the list, and extract .txt track files for each track
    for k in range(0,len(lst)):
        df = pd.DataFrame(lst[k])
        num = k + 1
        completeName = os.path.join(allTracksDir,'track%i.txt' % num)
        df.to_csv(completeName, index=False, header=['Frame_Number','X-coordinate','Y-coordinate'])

    # fill in missing frames with NaN values
        totalnumber = (lstnan[k][-1][0] + 1)
        missing = sorted(list(set(range(int(totalnumber))) - set(lstnan[k][:,0])))
        for elem in missing:
            lstnan[k] = np.insert(lstnan[k], elem, [[elem, nan, nan]], axis = 0)

    # a dictionary of index-0 pts from each track
    trackOrigins = {} 
    for index, track in enumerate(lstnan):
        trackOrigins[index] = track[0][1:]

    return lst, lstnan, trackOrigins


# ! ############################################################################################
# # For use from home computer, comment this out at school
# if __name__ == '__main__':
#     filename = r'C:/Users/vivty/OneDrive/Documents/Python Programs/RMSD_2D-master/Practice data/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
#     txy_pts, tracks = open_tracks(filename)

#     save_path = 'C:/temp'
#     minfrm = 20
#     lst, lstnan, trackOrigins = gen_indiv_tracks(save_path, minfrm)
# ! ############################################################################################


# ! ############################################################################################
# For use from school, comment this out at home
if __name__ == '__main__':
    filename = r'/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Piezo1 Trajectory for Analysis/2018_Nov_tirfm_tdtpiezo_5sec/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    txy_pts, tracks = open_tracks(filename)

    save_path = r'/home/vivek/Documents/Python Programs/Piezo1/temp_outputs'
    minfrm = 20
    lst, lstnan, trackOrigins = gen_indiv_tracks(save_path, minfrm, tracks, txy_pts)
# ! ############################################################################################