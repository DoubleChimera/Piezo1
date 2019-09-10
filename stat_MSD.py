import codecs
import json
import math
import os.path
import warnings
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class json_converter(object):
    def json_tracks_to_df(self, file_path):
        self.objLoad = codecs.open(file_path, "r", encoding="utf-8").read()
        self.lstnan = np.array(json.loads(self.objLoad))
        self.arrNan = np.array([np.array(track) for track in self.lstnan])

        lst_part, lst_frame, lst_x, lst_y = ([] for i in range(4))
        for particle, track in enumerate(self.arrNan):
            lst_part.extend([particle] * len(track))
            lst_frame.extend(np.ndarray.tolist(track[:, 0]))
            lst_x.extend(np.ndarray.tolist(track[:, 1]))
            lst_y.extend(np.ndarray.tolist(track[:, 2]))
        self.tracks_df = pd.DataFrame(
            {"particle": lst_part, "frame": lst_frame, "x": lst_x, "y": lst_y}
        )
        return self.tracks_df

    def MSD_df_to_json(self, savePath, df_MSD):
        self.df_MSD = df_MSD
        print(self.df_MSD.columns[0])
        # Make directory if it doesn't exist already
        outMSDdf_json = os.path.join(savePath, "Statistics/MSDs")
        if not os.path.exists(outMSDdf_json):
            os.makedirs(outMSDdf_json)
        # Determine which MSD, name .json accordingly
        elif self.df_MSD.columns[0] == 0:
            outJsonName = os.path.join(outMSDdf_json, "TAMSD.json")
            # Output dataframe to .json in determined directory
            self.df_MSD.reset_index().to_json(outJsonName, orient="split")
        elif self.df_MSD.columns[0] == "<x>":
            outJsonName = os.path.join(outMSDdf_json, "EAMSD.json")
            # Output dataframe to .json in determined directory
            self.df_MSD.to_json(outJsonName, orient="split")
        elif self.df_MSD.columns[0] == "frame":
            outJsonName = os.path.join(outMSDdf_json, "All_Lagtimes.json")
            # Output dataframe to .json in determined directory
            self.df_MSD.to_json(outJsonName, orient="split")

    def MSD_json_to_df(self, jsonFilePath):
        self.filePath = jsonFilePath
        with open(self.filePath) as json_file:
            jdata = json.load(json_file)
        self.df_fromJson = pd.DataFrame(jdata)
        return self.df_fromJson


class stat_MSD(object):
    def pandas_concat(self, *args, **kwargs):
        kwargs.setdefault("sort", False)
        return pd.concat(*args, **kwargs)

    def pandas_rolling(self, df, window, *args, **kwargs):
        """ Use rolling to compute a rolling average
        """
        return df.rolling(window, *args, **kwargs).mean()

    def pandas_sort(self, df, by, *args, **kwargs):
        if df.index.name is not None and df.index.name in by:
            df.index.name += "_index"
        return df.sort_values(*args, by=by, **kwargs)

    def compute_drift(self, tracks, smoothing=0, pos_columns=["x", "y"]):
        """ Return the ensemble drift, xy(t)
        """
        f_sort = stat.pandas_sort(tracks, ["particle", "frame"])
        f_diff = f_sort[list(pos_columns) + ["particle", "frame"]].diff()

        f_diff.rename(columns={"frame": "frame_diff"}, inplace=True)
        f_diff["frame"] = f_sort["frame"]

        # Compute per frame averages and keep deltas of same particle
        # and b/w frames that are consecutive
        mask = (f_diff["particle"] == 0) & (f_diff["frame_diff"] == 1)
        dx = f_diff.loc[mask, pos_columns + ["frame"]].groupby("frame").mean()
        if smoothing > 0:
            dx = stat.pandas_rolling(dx, smoothing, min_periods=0)
        return dx.cumsum()

    def msd_N(self, N, t):
        """Computes the effective number of statistically independent measurements of
           the mean square displacement of a single trajectory
        """

        t = np.array(t, dtype=np.float)

        return np.where(
            t > N / 2,
            1
            / (
                1
                + ((N - t) ** 3 + 5 * t - 4 * (N - t) ** 2 * t - N)
                / (6 * (N - t) * t ** 2)
            ),
            6 * (N - t) ** 2 * t / (2 * N - t + 4 * N * t ** 2 - 5 * t ** 3),
        )

    def msd_iter(self, pos, lagtimes, result_columns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        allLags_DF = pd.DataFrame()
        allLags_List = []
        for lt in lagtimes:
            # ! Insert an if statement for deciding whether we will have the all tracks output or not
            # ! Toggle a boolean here....
            diff = pos[lt:] - pos[:-lt]
            diff_DF = pd.DataFrame({f"x_lag{lt}": diff[:, 0], f"y_lag{lt}": diff[:, 1]})
            allLags_DF = pd.concat([allLags_DF, diff_DF], axis=1)
            diff_List = allLags_DF.loc[:, [f"x_lag{lt}", f"y_lag{lt}"]]
            diffMean_List = np.nanmean(diff_List, axis=0)
            diffMeanSquared_List = np.nanmean(diff_List ** 2, axis=0)
            diffConcat_List = np.concatenate((diffMean_List, diffMeanSquared_List))
            if len(allLags_List) == 0:
                allLags_List = diffConcat_List
            else:
                allLags_List = np.vstack((allLags_List, diffConcat_List))
        results = pd.DataFrame(allLags_List, columns=result_columns, index=lagtimes)
        return results, allLags_DF

    def msdNan(
        self,
        track,
        pixelWidth,
        frameTime,
        max_lagtime=100,
        pos_columns=None,
        detail=True,
        outputAllLags=False,
    ):
        """ Compute the mean displacement and mean squared displacement
            of one trajectory over a range of time intervals.
        """
        self.track = track
        if pos_columns is None:
            pos_columns = ["x", "y"]
        result_columns = ["<{}>".format(p) for p in pos_columns] + [
            "<{}^2>".format(p) for p in pos_columns
        ]

        try:
            self.pos = self.track.set_index("frame")[pos_columns] * pixelWidth
            self.pos = self.pos.reindex(
                np.arange(self.pos.index[0], 1 + self.pos.index[-1])
            )
        except ValueError:
            if track["frame"].nunique() != len(self.track["frame"]):
                raise Exception(
                    "Cannot use msdNan, more than one trajectory " "per particle found."
                )
            else:
                raise

        # checking to be safe
        max_lagtime = min(max_lagtime, len(self.pos) - 1)

        lagtimes = np.arange(1, max_lagtime + 1)

        # ! if statement changing output here
        results, allLags_DF = stat.msd_iter(self.pos.values, lagtimes, result_columns)

        results["msd"] = results[result_columns[-len(pos_columns) :]].sum(1)
        if detail:
            # effective number of measurements
            # approximately corrected with number of gaps
            results["N"] = stat.msd_N(len(self.pos), lagtimes) * (
                len(self.track) / len(self.pos)
            )
        results["lagt"] = results.index.values / float(frameTime)
        results.index.name = "lagt"
        if outputAllLags:
            return results, allLags_DF
        else:
            del allLags_DF
            return results

    def genLagColumns(self, lag_columns, pos_columns):
        self.lag_columns = lag_columns
        self.pos_columns = pos_columns
        for lag in self.lag_columns:
            for p in self.pos_columns:
                yield "".join(map(str, (p, lag)))

    def indiv_msd(
        self,
        tracks,
        pixelWidth,
        frameTime,
        max_lagtime=100,
        statistic="msd",
        pos_columns=None,
    ):
        self.ids = []
        self.msds = []
        self.frameTime = frameTime
        self.pos_columns = pos_columns
        self.tracks = tracks
        allLagsOutput_DF = tracks

        if self.pos_columns is None:
            self.pos_columns = ["x", "y"]

        # Determines max track length and gens maxLagtimes List
        # Also gens all Lag Times dataframe
        self.maxLagTime = list(range(self.tracks.groupby("particle").count().max()[0]))
        # Inserts lagtimes into Dataframe
        allLagsOutput_DF.insert(2, "lagt", (self.tracks["frame"] * self.frameTime))
        self.lag_columns = ["_lag{}".format(l) for l in self.maxLagTime]
        self.lag_results = list(stat.genLagColumns(self.lag_columns, self.pos_columns))
        allLagsOutput_DF = allLagsOutput_DF.reindex(
            columns=allLagsOutput_DF.columns.tolist() + self.lag_results
        )
        allLagsOutput_DF.set_index("particle", inplace=True)
        for particle, track in self.tracks.groupby("particle"):
            results, allLags_DF = stat.msdNan(
                track,
                pixelWidth,
                frameTime,
                max_lagtime,
                pos_columns,
                detail=True,
                outputAllLags=True,
            )
            allLagsOutput_DF.loc[particle, [f"x_lag0", f"y_lag0"]] = track[
                self.pos_columns
            ].values
            lagNames = list(allLags_DF.columns)
            allLags_DF = allLags_DF.append(pd.Series(), ignore_index=True)
            allLagsOutput_DF.loc[particle, lagNames] = allLags_DF[lagNames].values
            allLagsOutput_DF.x *= pixelWidth
            allLagsOutput_DF.y *= pixelWidth
            allLagsOutput_DF.x_lag0 *= pixelWidth
            allLagsOutput_DF.y_lag0 *= pixelWidth
            self.msds.append(results)
            self.ids.append(particle)
        results = stat.pandas_concat(self.msds, keys=self.ids)
        results = results.swaplevel(0, 1)[statistic].unstack()
        lagt = results.index.values.astype("float64") / float(frameTime)
        results.set_index(lagt, inplace=True)
        results.index.name = "lagt"
        return results, allLagsOutput_DF

    def ensa_msd(
        self,
        tracks,
        pixelWidth,
        frameTime,
        max_lagtime=100,
        detail=True,
        pos_columns=None,
    ):
        """Compute the ensemble mean squared displacement of many particles
        """
        ids = []
        msds = []
        self.tracks = tracks

        for particle, track in self.tracks.reset_index(drop=True).groupby("particle"):
            msds.append(
                stat.msdNan(track, pixelWidth, frameTime, max_lagtime, pos_columns)
            )
            ids.append(particle)
        msds = stat.pandas_concat(msds, keys=ids, names=["particle", "frame"])
        results = msds.mul(msds["N"], axis=0).mean(level=1)
        # results_stderr = results.div(msds['N'].sem(level=1), axis=0)  # ! USE THIS FOR STANDARD ERROR - MAKING THE CLOUD :)
        results = results.div(msds["N"].mean(level=1), axis=0)
        if not detail:
            return results.set_index("lagt")["msd"]
        results["N"] = msds["N"].sum(level=1)
        return results


class plot_MSD(object):
    def plot_TAMSD_bestFit(self, msds, fit_range):
        self.fit_range = fit_range
        self.msds_vals = msds
        self.msds_vals = self.msds_vals.reset_index(name="Avg_TAMSD")
        self.slope, self.intercept = np.polyfit(
            np.log(self.msds_vals["lagt"][self.fit_range[0] : self.fit_range[1]]),
            np.log(self.msds_vals["Avg_TAMSD"][self.fit_range[0] : self.fit_range[1]]),
            1,
        )
        y_fit = np.exp(
            self.slope
            * np.log(self.msds_vals["lagt"][self.fit_range[0] : self.fit_range[1]])
            + self.intercept
        )
        self.line = pd.DataFrame({"lagt": self.msds_vals["lagt"], "Avg_TAMSD": y_fit})
        return self.line, self.slope, self.intercept

    def plot_TAMSD(self, indiv_msds, fit_range):
        self.indiv_msds = indiv_msds
        self.fit_range = fit_range
        # get half the track lengths
        self.indiv_msds_range = int(math.floor(self.indiv_msds.count().max() / 2))
        self.half_indices = self.indiv_msds.index[0 : self.indiv_msds_range]
        self.half_indiv_msds = pd.DataFrame(index=self.half_indices)
        for track in self.indiv_msds:
            half_last_index = (
                round((self.indiv_msds[track].last_valid_index() / 2) / 0.05) * 0.05
            )
            self.half_indiv_msds = self.half_indiv_msds.join(
                self.indiv_msds[track][0:half_last_index]
            )

        # Average track of all tracks
        self.avg_half_msd = self.half_indiv_msds.mean(axis=1)

        # plot results as half track lengths
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot individual tracks, set label for legend
        ax.plot(
            self.half_indiv_msds.index.values,
            self.half_indiv_msds.values,
            "k-",
            alpha=0.2,
            label="Individual Tracks",
        )
        # Plot the averaged track, set label for legend
        ax.plot(
            self.avg_half_msd.index.values,
            self.avg_half_msd.values,
            "b-",
            alpha=1,
            linewidth=3,
            label="Averaged Track",
        )
        # Linear Fit of Averaged Track
        self.line, self.slope, self.intercept = plot_MSD.plot_TAMSD_bestFit(
            self, self.avg_half_msd, self.fit_range
        )
        ax.plot(
            self.line["lagt"],
            self.line["Avg_TAMSD"],
            "-r",
            linewidth=3,
            label="Linear Fit: y = {:.2f} x + {:.2f}".format(
                self.slope, self.intercept
            ),
        )
        # Set the scale of the axes to 'log'
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Set the window title
        fig = plt.gcf()
        fig.canvas.set_window_title("Time-Averaged MSD")
        # Set the legend to show only one entry for indiv tracks
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=12)
        # Set the headline/title for the plot
        fig.suptitle("TAMSD: Average and Individual Tracks", fontsize=20)
        # Set the axes labels
        ax.set_ylabel(r"$\overline{\delta^2 (\Delta)}$  [$\mu$m$^2$]", fontsize=15)
        ax.set_xlabel("lag times [$s$]", fontsize=15)
        # Position the axes labels
        ax.xaxis.set_label_coords(0.5, -0.07)
        # Determine the min/max values for the x, y axes
        # Padding value to increase axes by
        axes_padding = 0.1
        # Calculate min/max values for axes
        self.x_min = self.avg_half_msd.index[0] - (
            self.avg_half_msd.index[0] * axes_padding
        )
        self.x_max = self.avg_half_msd.index.max() + (
            self.avg_half_msd.index.max() * axes_padding
        )
        self.y_min = self.half_indiv_msds.min().min() - (
            self.half_indiv_msds.min().min() * axes_padding
        )
        self.y_max = self.half_indiv_msds.max().max() + (
            self.half_indiv_msds.max().max() * axes_padding
        )
        # Set the min/max values for the x, y axes
        ax.set(ylim=(self.y_min, self.y_max), xlim=(self.x_min, self.x_max))
        # Display the TAMSD plot
        plt.show()

    def plot_EAMSD_bestFit(self, msds, fit_range):
        self.fit_range = fit_range
        self.msds_vals = msds
        self.slope, self.intercept = np.polyfit(
            np.log(self.msds_vals["lagt"][self.fit_range[0] : self.fit_range[1]]),
            np.log(self.msds_vals["msd"][self.fit_range[0] : self.fit_range[1]]),
            1,
        )
        y_fit = np.exp(
            self.slope
            * np.log(self.msds_vals["lagt"][self.fit_range[0] : self.fit_range[1]])
            + self.intercept
        )
        self.line = pd.DataFrame({"lagt": self.msds_vals["lagt"], "Avg_EAMSD": y_fit})
        return self.line, self.slope, self.intercept

    def plot_EAMSD(self, ensa_msds, fit_range):
        self.ensa_msds = ensa_msds
        self.fit_range = fit_range
        # Plot results as half the track lengths by modifiying plotting window
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot EAMSD of tracks
        ax.plot(
            self.ensa_msds["lagt"],
            self.ensa_msds["msd"],
            "o",
            label="Ensemble Average MSD",
        )
        # Determine linear fit to data
        # Set number of initial points to fit
        # Fit from EAMSD calcs
        self.line, self.slope, self.intercept = plot_MSD.plot_EAMSD_bestFit(
            self, self.ensa_msds, self.fit_range
        )
        # Plot linear fit of EAMSD data
        ax.plot(
            self.line["lagt"],
            self.line["Avg_EAMSD"],
            "-r",
            linewidth=3,
            label="Linear Fit: y = {:.2f} x + {:.2f}".format(
                self.slope, self.intercept
            ),
        )
        # Plot error as a cloud around linear fit # ! Not implemented
        # Set the scale of the axes to 'log'
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Set the window title
        fig = plt.gcf()
        fig.canvas.set_window_title("Ensemble-Averaged MSD")
        # Set the legend
        ax.legend(loc="upper left", fontsize=12)
        # Set the headline/title for the plot
        fig.suptitle("Ensemble-Averaged MSD with a Linear Fit", fontsize=20)
        # Set the axes labels
        ax.set_ylabel(
            r"$\langle$$\bf{r}$$^2$($\Delta)\rangle$ [$\mu$m$^2$]", fontsize=15
        )
        ax.set_xlabel("lag time [$s$]", fontsize=15)
        # Position the axes labels
        ax.xaxis.set_label_coords(0.5, -0.07)
        # Determine the min/max values for the x, y axes
        # Padding value to increase axes by
        axes_padding = 0.1
        # Calculate min/max values for axes
        self.x_min = self.ensa_msds["lagt"].min() - (
            self.ensa_msds["lagt"].min() * axes_padding
        )
        self.x_max = self.ensa_msds["lagt"].max() + (
            self.ensa_msds["lagt"].max() * axes_padding
        )
        self.y_min = self.ensa_msds["msd"].min() - (
            self.ensa_msds["msd"].min() * axes_padding
        )
        self.y_max = self.ensa_msds["msd"].max() + (
            self.ensa_msds["msd"].max() * axes_padding
        )
        # Set the min/max values for x, y axes
        ax.set(ylim=(self.y_min, self.y_max), xlim=(self.x_min, self.x_max))
        # Display the EAMSD plot
        plt.show()


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
    jsonTracksLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"
    savePath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs"

    # in micros
    pixelWidth = 0.1092

    # milliseconds between frames from experiment, typically 50ms or 100ms
    frameTime = 100

    # Generates all the lagtimes for each track *VERY TIME INTENSIVE*
    genAllLagsOutput = False

    # bounding indices for linear fit
    fit_range = [1, 25]
    # * -----END OF USER INPUTS----- * #

    frameTime = frameTime / 1000  # Converts frame time to frames-per-second

    # Instantiate the json_converter class
    jc = json_converter()

    # Store tracks data into a pandas DataFrame
    tracks = jc.json_tracks_to_df(jsonTracksLoadPath)

    # Instantiate the stat_MSD class
    stat = stat_MSD()

    # Instantiate the plot_MSD class
    pMSD = plot_MSD()

    # * Ensemble average mean squared displacement calculation
    # Get the ensemble msd trajectory
    ensa_msds = stat.ensa_msd(tracks, pixelWidth, frameTime)
    # Output EAMSD .json to savePath
    jc.MSD_df_to_json(savePath, ensa_msds)

    # * Time-averaged mean squared displacement calculation
    # Get the individual trajectories
    indiv_msds, allLagTimes = stat.indiv_msd(tracks, pixelWidth, frameTime)
    # Output TAMSD.json to savePath
    jc.MSD_df_to_json(savePath, indiv_msds)
    jc.MSD_df_to_json(savePath, allLagTimes)

    # * TAMSD and EAMSD Plots
    # Plot TAMSD
    pMSD.plot_TAMSD(indiv_msds, fit_range)

    # Plot EAMSD
    pMSD.plot_EAMSD(ensa_msds, fit_range)
