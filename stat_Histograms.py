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
    pos_columns = None
    if pos_columns is None:
        pos_columns = ['x', 'y']


    def genLagColumns(lag_columns, pos_columns):
        for lag in lag_columns:
            for p in pos_columns:
                yield ''.join(map(str, (p, lag)))

    # Generates all displacement for a single track and returns the resulting DF
    def indivTrackDispls(track, pixelWidth, frameTime, maxLagTime, pos_columns=None):
        if pos_columns is None:
            pos_columns = ['x', 'y']
        # Make a list of tracks and determine the max track length
        for lag in range(int(track['frame'].max())):
            return None

    # Sets the index of tracks DF to 'particle' # ! Unnecessary?
    # tracks.set_index('particle', inplace=True)

    # Determines max track length and gens maxLagtimes List
    maxLagTime = list(range(tracks.groupby('particle').count().max()[0]))
    # Determines total number of particles in DF
    totalParticleCount = tracks.groupby('frame').count().max()[0]

    # self.pos = self.pos.reindex(np.arange(self.pos.index[0],
    #                                       1 + self.pos.index[-1]))

    # ! THIS WORKS FOR A SINGLE TRAJECTORY
    # for particle, track in tracks.reset_index(drop=True).groupby('particle'):
    #     # Gen lagt column and insert in 2nd position
    #     track.insert(2, 'lagt', (track['frame'] * frameTime), True)
    #     # Gen suffix of lag columns
    #     lag_columns = ['_lag{}'.format(l) for l in maxLagTime]
    #     # Use pos_columns to gen prefix of lag columns
    #     lag_results = list(genLagColumns(lag_columns, pos_columns))
    #     # Add the lag columns to what will be the output results df
    #     results = track.reindex(columns = track.columns.tolist() + lag_results)
    #     # run some function that takes in a single track and calcs disps
    #     trackDisps = indivTrackDispls(track, pixelWidth, frameTime, maxLagTime)
    #     # results = results.append(trackDisps, sort=False)

    #     for lag in range(int(len(track['frame']))):
    #         if lag == 0:
    #             results[[f'x_lag{lag}', f'y_lag{lag}']] = track[pos_columns]
    #         else:
    #             indivDisp = track[pos_columns].values[lag:] - track[pos_columns].values[:-lag]
    #             results[[f'x_lag{lag}']] = pd.Series(indivDisp[:,0])
    #             results[[f'y_lag{lag}']] = pd.Series(indivDisp[:,1])
    # print(results)
    # ! END OF SINGLE TRAJECTORY SUCCESS STORY
    # ? test
    # print(tracks.set_index('particle').loc[0])
    # ? End test

    tracks.insert(2, 'lagt', (tracks['frame'] * frameTime))
    lag_columns = ['_lag{}'.format(l) for l in maxLagTime]
    lag_results = list(genLagColumns(lag_columns, pos_columns))
    results = tracks.reindex(columns = tracks.columns.tolist() + lag_results)
    results.set_index('particle', inplace=True)
    # print(results.loc[0])
    # results.loc[0][['x_lag99', 'y_lag99']] = [[35, 35]]

    for particle, track in tracks.reset_index(drop=True).groupby('particle'):
        trackLength = int(len(track['frame']))
        for lag in range(trackLength):
            if lag == 0:
                results.loc[particle][[f'x_lag{lag}',f'y_lag{lag}']] = track[pos_columns].values
            else:
                indivDisp = pd.DataFrame(track[pos_columns].values[lag:] - track[pos_columns].values[:-lag],
                                         columns=[[f'x_lag{lag}', f'y_lag{lag}']])
                # ! HERE I AM TRYING TO GET THIS TO MERGE INTO THE RESULTS
                # ! Use a loop to figure out how many rows are missing
                # ! add np.nan values there
                # ! combine with results
                print(results)
                print(indivDisp)
                results = pd.concat([results.loc[particle], indivDisp], sort=False)
                # results.loc[particle][[f'x_lag{lag}']] = pd.Series(indivDisp[:,0])
                # results.loc[particle][[f'y_lag{lag}']] = pd.Series(indivDisp[:,1])


        # msds.append(stat.msdNan(track, pixelWidth, frameTime, max_lagtime, pos_columns))
        # ids.append(particle)


    # takes a track and changes the pixels to microns # ! Keepable
    # pos = track.set_index('frame')[pos_columns] * pixelWidth


    # Take in tracks
    # Adjust values with pixelWidth so distances are in microns
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
