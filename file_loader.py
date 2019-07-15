# -*- coding: utf-8 -*-

# todo  Step 1:     Load a .tif, grab the first frame and plot on top
# todo              ask for initial parameters about experiement
# todo
# ?                 What has been done?
# // DONE             A .json is loaded into a list of arrays corresponding to each track
# // DONE             --Now need to fill in skipped frames with NaN values

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


def gen_indiv_tracks(save_path, minfrm):
    """
    returns lst[] and lstnan[]
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

    lstnan = np.copy(lst)

    # parse through the list, and extract .txt track files for each track
    for k in range(0,len(lst)):
        df = pd.DataFrame(lst[k])
        num = k + 1
        completeName = os.path.join(save_path,'track%i.txt' % num)
        df.to_csv(completeName, index=False, header=['Frame_Number','X-coordinate','Y-coordinate'])

    # fill in missing frames with NaN values 
        totalnumber = (lstnan[k][-1][0] + 1)
        missing = sorted(list(set(range(int(totalnumber))) - set(lstnan[k][:,0])))
        for elem in missing:
            lstnan[k] = np.insert(lstnan[k], elem, [[elem, nan, nan]], axis = 0)
    return lst, lstnan

def rmsd()

    return lstrmsd


if __name__ == '__main__':
    filename = r'C:/Users/vivty/OneDrive/Documents/Python Programs/RMSD_2D-master/Practice data/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    txy_pts, tracks = open_tracks(filename)

    save_path = 'C:/temp'
    minfrm = 20
    lst, lstnan = gen_indiv_tracks(save_path, minfrm)


# * Current Debugging code begins below this point
# * ----------------------------------------------------------------------------

# print(lst[0])
# print(lstnan[0])

# * ----------------------------------------------------------------------------

# ! ----------------------------------------------------------------------------
# ! Old Debugging code begins below this point
# ! ----------------------------------------------------------------------------
# i = 100
# print(lstnan[i])
# print(lst[i])
# print("Length of original = {}".format(str(len(lst[i]))))
# print("Length of adjusted = {}".format(str(len(lstnan[i]))))

# nan = np.nan                                                    # ! done
# a = lst                                                         # x not needed already there
# i = 1                                                           # x not needed already there
# listlength = len(lst[i])                                        # x not needed, unecessary
# totalnumber = (lst[i][-1][0] + 1)                               # ! done
# print(lst[i])                                                   # x not needed, unecessary
# print(str(totalnumber))                                         # x not needed, unecessary
# print("Length = {}".format(str(listlength)))                    # x not needed, unecessary
# print("Difference = {}".format(str(totalnumber - listlength)))  # x not needed, unecessary
# print(lst[i][:,0])                                              # x not needed, unecessary
# print("-----------")                                            # x not needed, unecessary
# # make a set that has the same range as                         # x not needed, unecessary
# # the final index of this one                                   # x not needed, unecessary
# # compare the sets and pull out differences                     # x not needed, unecessary
# missing = list(set(range(int(totalnumber))) - set(lst[i][:,0])) # ! done
# print("Missing = {}".format(str(missing)))                      # x not needed, unecessary
# # x in track1, point index 85 is missing                        # x not needed, unecessary
# # x and needs to be replaced with a NaN value                   # x not needed, unecessary
# for index, elem in enumerate(sorted(missing)):                  # ! done
#     a[i] = np.insert(a[i], elem, [[elem, nan, nan]], axis = 0)  # ! done
#     print(a[i]) 
# print("-------------------------------------------")
# print(a[i])
# print(a[i])
# if the length of the array is the same as the last indexed value, no points were skipped,
# and no edits need to be made
# if the length differs, we need to find the inconsistent points and insert a NaN value
# start with a for loop going over each array, check the length with index, if same, skip
# if not same, parse over, find inconsistency and add missing frame value and NaN for x, y

#  list name, index, array, axis
# np.insert(a, 3, [[2, 2]], axis = 0)b
