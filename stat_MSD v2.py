import codecs
import warnings
from warnings import warn
import six
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

class json_track_loader(object):
    def json_to_dataframe(self, file_path):
        self.objLoad = codecs.open(file_path, 'r', encoding='utf-8').read()
        self.lstnan = np.array(json.loads(self.objLoad))
        self.arrNan = np.array([np.array(track) for track in self.lstnan])

        lst_part, lst_frame, lst_x, lst_y = ([] for i in range(4))
        for particle, track in enumerate(self.arrNan):
            lst_part.extend([particle]*len(track))
            lst_frame.extend(np.ndarray.tolist(track[:,0]))
            lst_x.extend(np.ndarray.tolist(track[:,1]))
            lst_y.extend(np.ndarray.tolist(track[:,2]))
        self.tracks_df = pd.DataFrame({'particle':lst_part, 'frame':lst_frame, 'x':lst_x, 'y':lst_y})
        return self.tracks_df

    def pandas_concat(self, *args, **kwargs):
        kwargs.setdefault('sort', False)
        return pd.concat(*args, **kwargs)

    def msd_N(self, N, t):
        """Computes the effective number of statistically independent measurements of 
           the mean square displacement of a single trajectory
        """

        t = np.array(t, dtype=np.float)
        return np.where(t > N/2, 
                        1/(1+((N-t)**3+5*t-4*(N-t)**2*t-N)/(6*(N-t)*t**2)),
                        6*(N-t)**2*t/(2*N-t+4*N*t**2-5*t**3))

    def msd_iter(self, pos, lagtimes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        for lt in lagtimes:
            diff = pos[lt:] - pos[:-lt]
            yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff**2, axis=0)))

    def msdNan(self, track, pixelWidth, frameTime, max_lagtime=100, pos_columns=None, detail=True):
        """ Compute the mean displacement and mean squared displacement of one trajectory over a 
            range of time intervals.
        """
        self.track = track
        if pos_columns is None:
            pos_columns = ['x', 'y']
        result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                         ['<{}^2>'.format(p) for p in pos_columns]

        try:
            self.pos = self.track.set_index('frame')[pos_columns] * pixelWidth
            self.pos = self.pos.reindex(np.arange(self.pos.index[0], 1 + self.pos.index[-1]))
        except ValueError:
            if track['frame'].nunique()!=len(self.track['frame']):
                raise Exception("Cannot use msdNan, more than one trajectory "
                                "per particle found.")
            else:
                raise

        max_lagtime = min(max_lagtime, len(self.pos) - 1) # checking to be safe

        lagtimes = np.arange(1, max_lagtime + 1)

        results = pd.DataFrame(jtl.msd_iter(self.pos.values, lagtimes), columns=result_columns, index=lagtimes)

        results['msd'] = results[result_columns[-len(pos_columns):]].sum(1)
        if detail:
            # effective number of measurements
            # approximately corrected with number of gaps
            results['N'] = jtl.msd_N(len(self.pos), lagtimes) * (len(self.track) / len(self.pos))
        results['lagt'] = results.index.values/float(frameTime)
        results.index.name = 'lagt'
        return results

    def indiv_msd(self, tracks, pixelWidth, frameTime, max_lagtime=100, statistic='msd', pos_columns=None):
        self.ids = []
        self.msds = []
        self.tracks = tracks
        for particle, track in self.tracks.groupby('particle'):
            self.msds.append(jtl.msdNan(track, pixelWidth, frameTime, max_lagtime, pos_columns, detail=True))
            self.ids.append(particle)
        results = jtl.pandas_concat(self.msds, keys=self.ids)
        results = results.swaplevel(0, 1)[statistic].unstack()
        lagt = results.index.values.astype('float64')/float(frameTime)
        results.set_index(lagt, inplace=True)
        results.index.name = 'lag time [s]'
        return results

    def ensa_msd(self, tracks, pixelWidth, frameTime, max_lagtime=100, detail=True, pos_columns=None):
        """Compute the ensemble mean squared displacement of many particles
        """
        ids = []
        msds = []
        self.tracks = tracks
        for particle, track in self.tracks.reset_index(drop=True).groupby('particle'):
            msds.append(jtl.msdNan(track, pixelWidth, frameTime, max_lagtime, pos_columns))
            ids.append(particle)
        msds = jtl.pandas_concat(msds, keys=ids, names=['particle', 'frame'])
        results = msds.mul(msds['N'], axis=0).mean(level=1)
        results = results.div(msds['N'].mean(level=1), axis=0)
        if not detail:
            return results.set_index('lagt')['msd']
        results['N'] = msds['N'].sum(level=1)
        return results


if __name__ == '__main__':
    #################### * USER INPUTS BELOW * ####################
    fileLoadPath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp/Selected_tracks/selected_track_list.json'
    savePath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    # time (in ms) between frames from experiment, typically 50ms or 100ms
    pixelWidth = .1092      # in microns
    frameTime = 50          # in milliseconds

    #################### * END OF USER INPUTS * ###################
    frameTime = 1000 / frameTime  # Converts frame time to frames-per-second
    jtl = json_track_loader()
    tracks = jtl.json_to_dataframe(fileLoadPath)


    # * #################### CURRENT DEBUGGING CODE IS BELOW ####################
    print(tracks)
    results = jtl.ensa_msd(tracks, pixelWidth, frameTime)
    print(results)


    # ! ####################   OLD DEBUGGING CODE IS BELOW   ####################
    # # Set 'r' as trackArray
    # r = trackArray # * DONE!
    # # make a copy to work with of trackArray, store as 's'   # * DO WE NEED THIS COPY????  NO!!!  DONE!
    # s = r.copy() # * DONE! IGNORED!
    # # find the length of the longest track in trackArray
    # maxVal = 0  # * DONE!
    # # this returns a list of the lengths of all the tracks
    # maxVal = [val.shape[0] for val in s if val.shape[0] > maxVal] # * DONE!
    # # make a list of zeros as long as the longest track with 2 columns for x-y
    # a = np.zeros((max(maxVal), 1), dtype='float') # * DONE!
    # # print(a)

    # # for each track
    # for track in s: # * DONE!
    # # subtrack the origin val from each frame
    #     diff = track[:,1:3] - track[0,1:3] # * DONE!
    # # square those values and sum each row
    #     diff = np.square(diff).sum(axis=1).reshape(len(diff),1) # * DONE!
    # # add thos values to the total msd matrix, column 1
    #     b = np.sum(np.stack((a[:len(diff)], diff)), axis=0) # * DONE!
    #     a[:len(diff)] = b # * DONE!
    # # divide by the number of tracks
    # a = a / len(s) # * DONE!

    # # Now add in the lag times...
    # # first assume a time per frame of 50 ms
    # frame_time = .050 # * DONE!
    # timing = (np.arange(0, len(a), dtype='float').reshape(100,1)) * frame_time # * DONE!
    # a = np.append(timing, a, axis = 1) # * DONE!
    # a = pd.DataFrame(a) # * DONE!
    # a.columns = ["Lag Time","EAMSD"] # * DONE!
    # # a.set_index('Lag Time', drop=True, inplace=True)  # * Unused - sets an arbitrary column as the index, in this case 'Lag Time'
    # a.dropna(inplace=True) # * DONE!

    # print(s[0][0,1:3])
    # print(s[0][0])
    # diff = s[0][:,1:3] - s[0][0,1:3]
    # print(diff)
    # diff = np.square(diff).sum(axis=1)
    # print(diff)
    # print(type(diff))


    # for val in s:
    #     if val.shape[0] > max:
    #         max = val.shape[0]
    #     print(max)

    # Tracks can be accessed by index, i.e. trackArray[0], trackArray[1], etc.
    # Frame index: trackArray[0][:,0]    X-coords: trackArray[0][:,1]   Y-coords: trackArray[0][:,2]  XY-coords: trackArray[0][:,1:3]
    # XY-coord of 0th frame trackArray[0][0,1:3]