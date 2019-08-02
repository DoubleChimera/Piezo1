# Calcs statistics - MSD

import codecs
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as stats
from collections import OrderedDict


class json_track_loader(object):
    def json_to_dataframe(self, file_path):
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


class stat_MSD(object):
    def pandas_concat(self, *args, **kwargs):
        kwargs.setdefault('sort', False)
        return pd.concat(*args, **kwargs)

    def pandas_rolling(self, df, window, *args, **kwargs):
        """ Use rolling to compute a rolling average
        """
        return df.rolling(window, *args, **kwargs).mean()

    def pandas_sort(self, df, by, *args, **kwargs):
        if df.index.name is not None and df.index.name in by:
            df.index.name += '_index'
        return df.sort_values(*args, by=by, **kwargs)

    def compute_drift(self, tracks, smoothing=0, pos_columns=['x', 'y']):
        """ Return the ensemble drift, xy(t)
        """
        f_sort = stat.pandas_sort(tracks, ['particle', 'frame'])
        f_diff = f_sort[list(pos_columns) + ['particle', 'frame']].diff()

        f_diff.rename(columns={'frame': 'frame_diff'}, inplace=True)
        f_diff['frame'] = f_sort['frame']

        # Compute per frame averages and keep deltas of same particle
        # and b/w frames that are consecutive
        mask = (f_diff['particle'] == 0) & (f_diff['frame_diff'] == 1)
        dx = f_diff.loc[mask, pos_columns + ['frame']].groupby('frame').mean()
        if smoothing > 0:
            dx = stat.pandas_rolling(dx, smoothing, min_periods=0)
        return dx.cumsum()

    def msd_N(self, N, t):
        """Computes the effective number of statistically independent measurements of
           the mean square displacement of a single trajectory
        """

        t = np.array(t, dtype=np.float)

        return np.where(t > N / 2,
                        1 / (1 + ((N - t) ** 3 + 5 * t - 4 * (N - t) ** 2 * t - N) / (6 * (N - t) * t ** 2)),
                        6 * (N - t) ** 2 * t / (2 * N - t + 4 * N * t ** 2 - 5 * t ** 3))

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
            if track['frame'].nunique() != len(self.track['frame']):
                raise Exception("Cannot use msdNan, more than one trajectory "
                                "per particle found.")
            else:
                raise

        # checking to be safe
        max_lagtime = min(max_lagtime, len(self.pos) - 1)

        lagtimes = np.arange(1, max_lagtime + 1)

        results = pd.DataFrame(stat.msd_iter(self.pos.values, lagtimes),
                               columns=result_columns, index=lagtimes)

        results['msd'] = results[result_columns[-len(pos_columns):]].sum(1)
        if detail:
            # effective number of measurements
            # approximately corrected with number of gaps
            results['N'] = stat.msd_N(len(self.pos), lagtimes) * (len(self.track) / len(self.pos))
        results['lagt'] = results.index.values / float(frameTime)
        results.index.name = 'lagt'
        return results

    def indiv_msd(self, tracks,
                  pixelWidth, frameTime,
                  max_lagtime=100, statistic='msd',
                  pos_columns=None):
        self.ids = []
        self.msds = []
        self.tracks = tracks
        for particle, track in self.tracks.groupby('particle'):
            self.msds.append(stat.msdNan(track, pixelWidth, frameTime,
                                         max_lagtime, pos_columns, detail=True))
            self.ids.append(particle)
        results = stat.pandas_concat(self.msds, keys=self.ids)
        results = results.swaplevel(0, 1)[statistic].unstack()
        lagt = results.index.values.astype('float64') / float(frameTime)
        results.set_index(lagt, inplace=True)
        results.index.name = 'lagt'
        return results

    def ensa_msd(self, tracks, pixelWidth, frameTime,
                 max_lagtime=100, detail=True, pos_columns=None):
        """Compute the ensemble mean squared displacement of many particles
        """
        ids = []
        msds = []
        self.tracks = tracks
        for particle, track in self.tracks.reset_index(drop=True).groupby('particle'):
            msds.append(stat.msdNan(track, pixelWidth, frameTime, max_lagtime, pos_columns))
            ids.append(particle)
        msds = stat.pandas_concat(msds, keys=ids, names=['particle', 'frame'])
        results = msds.mul(msds['N'], axis=0).mean(level=1)
        # results_stderr = results.div(msds['N'].sem(level=1), axis=0)  # ! USE THIS FOR STANDARD ERROR - MAKING THE CLOUD :)
        results = results.div(msds['N'].mean(level=1), axis=0)
        if not detail:
            return results.set_index('lagt')['msd']
        results['N'] = msds['N'].sum(level=1)
        return results


class plot_MSD(object):

    def plot_TAMSD(self, indiv_msds):
        self.indiv_msds = indiv_msds
        # get half the track lengths
        self.indiv_msds_range = (int(math.floor(self.indiv_msds.count().max() / 2)))
        self.half_indices = self.indiv_msds.index[0:self.indiv_msds_range]
        self.half_indiv_msds = pd.DataFrame(index=self.half_indices)
        for track in self.indiv_msds:
            half_last_index = (round((self.indiv_msds[track].last_valid_index() / 2) / 0.05) * 0.05)
            self.half_indiv_msds = self.half_indiv_msds.join(self.indiv_msds[track][0:half_last_index])

        # Average track of all tracks
        self.avg_half_msd = self.half_indiv_msds.mean(axis=1)

        # plot results as half track lengths
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot individual tracks, set label for legend
        ax.plot(self.half_indiv_msds.index,
                self.half_indiv_msds,
                'k-',
                alpha=0.2,
                label='Individual Tracks')
        # Plot the averaged track, set label for legend
        ax.plot(self.avg_half_msd.index,
                self.avg_half_msd,
                'r-',
                alpha=1,
                linewidth=3,
                label='Averaged Track')
        # Set the scale of the axes to 'log'
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Set the window title
        fig = plt.gcf()
        fig.canvas.set_window_title('Time-Averaged MSD')
        # Set the legend to show only one entry for indiv tracks
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=12)
        # Set the headline/title for the plot
        fig.suptitle('TAMSD: Average and Individual Tracks', fontsize=20)
        # Set the axes labels
        ax.set_ylabel(r'$\overline{\delta^2 (\Delta)}$  [$\mu$m$^2$]', fontsize=15)
        ax.set_xlabel('lag times [$s$]', fontsize=15)
        # Position the axes labels
        ax.xaxis.set_label_coords(0.5, -0.07)
        # Display the TAMSD plot
        plt.show()

    def plot_EAMSD(self, ensa_msds):
        self.ensa_msds = ensa_msds
        # Plot results as half the track lengths by modifiying plotting window
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot EAMSD of tracks
        ax.plot(self.ensa_msds['lagt'],
                self.ensa_msds['msd'],
                'o',
                label="Ensemble Average MSD")
        # Determine linear fit to data
        # Set number of initial points to fit
        self.fit_range = 15
        (self.slope,
         self.intercept,
         self.r_value,
         self.p_value,
         self.std_err) = stats.linregress(self.ensa_msds['lagt'][0:self.fit_range],
                                          self.ensa_msds['msd'][0:self.fit_range])
        self.line = (self.slope * self.ensa_msds['lagt'] + self.intercept)
        self.line = pd.DataFrame({'lagt': self.ensa_msds['lagt'],
                                  'Avg_eamsd': self.line.values})
        # Plot linear fit of EAMSD data
        ax.plot(self.line['lagt'],
                self.line['Avg_eamsd'],
                '-r',
                linewidth=3,
                label='Linear Fit: y = {:.2f} x + {:.2f}'.format(self.slope, self.intercept))
        # Plot error as a cloud around linear fit # ! Not implemented
        # Set the scale of the axes to 'log'
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Set the window title
        fig = plt.gcf()
        fig.canvas.set_window_title('Ensemble-Averaged MSD')
        # Set the legend
        ax.legend(loc='upper left', fontsize=12)
        # Set the headline/title for the plot
        fig.suptitle('Ensemble-Averaged MSD with a Linear Fit', fontsize=20)
        # Set the axes labels
        ax.set_ylabel(r'$\langle$$\bf{r}$$^2$($\Delta)\rangle$ [$\mu$m$^2$]', fontsize=15)
        ax.set_xlabel('lag time [$s$]', fontsize=15)
        # Position the axes labels
        ax.xaxis.set_label_coords(0.5, -0.07)
        # Determine and set x-lim and y-lim of plot
        self.half_x_max = round((self.ensa_msds['lagt'].max() / 2) / 0.05) * 0.05
        self.x_min = 4e-2
        self.x_range = self.half_x_max - self.x_min
        self.y_min = 5e-3
        self.y_max = round((self.y_min + self.x_range - 2.3), 2)
        ax.set(ylim=(self.y_min, self.y_max), xlim=(self.x_min, self.half_x_max))
        # Display the EAMSD plot
        plt.show()


if __name__ == '__main__':
    #################### * USER INPUTS BELOW * ####################
    fileLoadPath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp/Selected_tracks/selected_track_list.json'
    savePath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    # time (in ms) between frames from experiment, typically 50ms or 100ms
    pixelWidth = .1092      # in microns
    frameTime = 50          # in milliseconds

    #################### * END OF USER INPUTS * ###################
    frameTime = 1000 / frameTime  # Converts frame time to frames-per-second

    # Instantiate the json_track_loader class
    jtl = json_track_loader()

    # Store data into a pandas DataFrame
    tracks = jtl.json_to_dataframe(fileLoadPath)

    # Instantiate the stat_MSD class
    stat = stat_MSD()

    # Instantiate the plot_MSD class
    pMSD = plot_MSD()

    # * Time-averaged mean squared displacement
    # Get the individual trajectories
    indiv_msds = stat.indiv_msd(tracks, pixelWidth, frameTime)

    # Plot TAMSD
    pMSD.plot_TAMSD(indiv_msds)

    # * Ensemble average mean squared displacement
    # Get the ensemble msd trajectory
    ensa_msds = stat.ensa_msd(tracks, pixelWidth, frameTime)

    # Plot EAMSD
    pMSD.plot_EAMSD(ensa_msds)

    # * #################### CURRENT DEBUGGING CODE IS BELOW ####################

    # ! plots need: legends, larger labels, titles, fitting-data, error clouds
    # ! Make a new class for plotting all the data

    # ! ####################   OLD DEBUGGING CODE IS BELOW   ####################
