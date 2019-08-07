import json
import codecs
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class json_converter(object):
    def json_tracks_to_df(self, file_path):
        self.objLoad = codecs.open(file_path, 'r', encoding='utf-8').read()
        self.lstnan = np.array(json.loads(self.objLoad))
        self.arrNan = np.array([np.array(track) for track in self.lstnan])

        lst_part, lst_frame, lst_x, lst_y = ([] for i in range(4))
        for particle, track in enumerate(self.arrNan):
            lst_part.extend([particle] * len(track))
            lst_frame.extend(np.ndarray.tolist(track[:, 0]))
            lst_x.extend(np.ndarray.tolist(track[:, 1]))
            lst_y.extend(np.ndarray.tolist(track[:, 2]))
        self.tracks_df = pd.DataFrame({'particle': lst_part,
                                       'frame': lst_frame,
                                       'x': lst_x,
                                       'y': lst_y})
        return self.tracks_df


if __name__ == '__main__':

    #################### * USER INPUTS BELOW * ####################
    savePath = r'/home/vivek/Documents/Python Programs/Piezo1/temp_outputs'
    jsonTracksLoadPath = r'/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json'
    loadPath = r'/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs'
    frameTime = 50          # in milliseconds
    pixelWidth = 0.1092     # in microns
    #################### * END OF USER INPUTS * ###################
    frameTime = frameTime / 1000
    # Instantiate json_converter() class
    jc = json_converter()
    # Pull tracks into dataframe from .json file
    # These tracks use pixels and frames, not microns and seconds
    tracks = jc.json_tracks_to_df(jsonTracksLoadPath)

    # ! #################### CURRENT DEBUGGING CODE IS BELOW ####################
    pos_columns=None
    if pos_columns is None:
        pos_columns = ['x', 'y']
    lagtimes = [1, 2, 3]
    lag_columns = ['_lag{}'.format(l) for l in lagtimes]
    def genLagColumns(lag_columns, pos_columns):
        for lag in lag_columns:
            for p in pos_columns:
                yield ''.join(map(str, (p, lag)))
    results = list(combineLags(lag_columns, pos_columns))

    def genDisplacements(tracks, pixelWidth, pos_columns=None):
        if pos_columns is None:
            pos_columns = ['x', 'y']
        return None

    # Take in tracks
    # Adjust values with pixelWidth
    # Determine the longest track length in tracks
    #   Use this to determine max columns of lag
    # Loop over each particles trajectory
    #   Store the intial trajectory coordinates in a new DF -columns ('x', 'y') with index frame
    #   Calculate an x Frame displacement for whole trajectory
    #   Append it to the list for that particle ('x_lag1', 'y_lag1')
    #       Continue this operation for the whole trajectory length
    #   Fill in any blanks in the dataframe with NaN values up to max columns of lag
    #   Continue this for all particles
    #   Put all particles and lags together in the same DF
    # Should end with a DF that has all particles and lagtimes
    # Output this as a .json file

    # TODO Make a function that takes in the tracks dataframe as an argument
    # TODO Loop over each particle-track, calculate displacements for all lag times
    # TODO Apppend the new list with new column labels to that particles portion of the dataframe
    # TODO Output that dataframe as a .json file
    # TODO Write a function to take in that .json file as a dataframe for future processing

    # ! ####################   OLD DEBUGGING CODE IS BELOW   ####################
