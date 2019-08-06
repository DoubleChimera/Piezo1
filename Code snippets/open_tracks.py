import numpy as np
import json, codecs

def open_tracks(filename):
""" -*- open_tracks() docstring and example -*-
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

if __name__ == '__main__':
    filename = r'/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Piezo1_Trajectory/2018_Nov_tirfm_tdtpiezo_5sec/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    txy_pts, tracks = open_tracks(filename)
