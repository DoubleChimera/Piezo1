import codecs
import json
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict

# import stat_MSD_outputs as statMSD


class json_converter(object):
    def json_SelectedTracks_to_DF(self, file_path):
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


def mobile_trapped_tracks(tracksDF, locErr, locErrLagTime, savePath):
    # Set up empty vars
    mobileTrappedTracksDict = {}
    trappedTracksList = []
    mobileTracksList = []

    # Iterate over list and separate tracks by cutoff values
    for particle, displacement in enumerate(tracksDF.loc[locErrLagTime]):
        if displacement <= locErr:
            trappedTracksList.append(particle)
        else:
            mobileTracksList.append(particle)

    # Store lists as a dictionary
    mobileTrappedTracksDict["Mobile"] = mobileTracksList
    mobileTrappedTracksDict["Trapped"] = trappedTracksList

    # Setup the output path and directory
    outMobileTrappedDict_json = os.path.join(savePath, "Statistics/MSDs")
    if not os.path.exists(outMobileTrappedDict_json):
        os.makedirs(outMobileTrappedDict_json)
    # Set the output name of the .json file
    outJsonName = os.path.join(outMobileTrappedDict_json, "Mobile_Trapped_tracks.json")
    # Output dict to .json in determined directory
    with open(outJsonName, "w") as filePath:
        json.dump(mobileTrappedTracksDict, filePath)

    # Return the generated dictionary
    return mobileTrappedTracksDict


def load_MobileTrapped_json(jsonFilePath):
    with open(jsonFilePath) as json_file:
        jdata = json.load(json_file)
    return jdata


def plot_TrappedTAMSD(TAMSD_DF, trappedTracks_List, frameTime):
    # Fix indices so that they are no longer randomized float values
    lagt = TAMSD_DF.index.values.astype("float64")
    lagt = np.round(lagt, 1)
    TAMSD_DF.set_index(lagt, inplace=True)
    TAMSD_DF.index.name = "lagt"
    # Setup a trappedTrack dataframe
    trappedTAMSDTracks_DF = TAMSD_DF.loc[:, trappedTracks_List]
    # get half the track lengths
    trappedTAMSD_range = int(math.floor(trappedTAMSDTracks_DF.count().max() / 2))
    trappedTAMSD_half_indices = np.round(
        trappedTAMSDTracks_DF.index[0:trappedTAMSD_range], 3
    )
    trappedTAMSD_half_MSDs = pd.DataFrame(index=trappedTAMSD_half_indices)
    for track in trappedTAMSDTracks_DF:
        half_last_index = round(
            (trappedTAMSDTracks_DF[track].last_valid_index() / 2) / (1 / frameTime)
        ) * (1 / frameTime)
        trappedTAMSD_half_MSDs[track] = trappedTAMSDTracks_DF[track][0:half_last_index]

    # Average track of all tracks
    avgTrappedTAMSD_half_MSD = trappedTAMSD_half_MSDs.mean(axis=1)

    # Plot results as half track lengths
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot individual tracks, set label for legend
    ax.plot(
        trappedTAMSD_half_MSDs.index,
        trappedTAMSD_half_MSDs,
        "k-",
        alpha=0.2,
        label="Trapped Tracks",
    )

    # Set the scale of the axes to 'log'
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Set the window title
    fig = plt.gcf()
    fig.canvas.set_window_title("TAMSD of Trapped Tracks")
    # Set the legend to show only one entry for tracks
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=12)
    # Set the headline/title for the plot
    fig.suptitle("TAMSD: Trapped Tracks", fontsize=20)
    # Set the axes labels
    ax.set_ylabel(r"$\overline{\delta^2 (\Delta)}$  [$\mu$m$^2$]", fontsize=15)
    ax.set_xlabel("lag times [$s$]", fontsize=15)
    # Position the axes labels
    ax.xaxis.set_label_coords(0.5, -0.07)
    # Determine the min/max values for the x, y axes
    # Padding value to increase axes by
    axes_padding = 0.1
    # Calculate min/max values for axes
    # Calculate min/max values for axes
    x_min = avgTrappedTAMSD_half_MSD.index[0] - (
        avgTrappedTAMSD_half_MSD.index[0] * axes_padding
    )
    x_max = avgTrappedTAMSD_half_MSD.index.max() + (
        avgTrappedTAMSD_half_MSD.index.max() * axes_padding
    )
    y_min = trappedTAMSD_half_MSDs.min().min() - (
        trappedTAMSD_half_MSDs.min().min() * axes_padding
    )
    y_max = trappedTAMSD_half_MSDs.max().max() + (
        trappedTAMSD_half_MSDs.max().max() * axes_padding
    )
    # Set the min/max values for the x, y axes
    ax.set(ylim=(y_min, y_max), xlim=(x_min, x_max))
    # Display the TAMSD plot
    plt.show()


def plot_TAMSD_bestFit(msds, fit_range):
    fit_range = fit_range
    msds_vals = msds
    msds_vals = msds_vals.reset_index(name="Avg_TAMSD")
    slope, intercept = np.polyfit(
        np.log(msds_vals["lagt"][fit_range[0] : fit_range[1]]),
        np.log(msds_vals["Avg_TAMSD"][fit_range[0] : fit_range[1]]),
        1,
    )
    y_fit = np.exp(
        slope * np.log(msds_vals["lagt"][fit_range[0] : fit_range[1]]) + intercept
    )
    line = pd.DataFrame({"lagt": msds_vals["lagt"], "Avg_TAMSD": y_fit})
    return line, slope, intercept


def plot_MobileTAMSD(TAMSD_DF, mobileTracks_List, frameTime, fit_range):
    # Fix indices so that they are no longer randomized float values
    lagt = TAMSD_DF.index.values.astype("float64")
    lagt = np.round(lagt, 1)
    TAMSD_DF.set_index(lagt, inplace=True)
    TAMSD_DF.index.name = "lagt"
    # Setup a trappedTrack dataframe
    mobileTAMSDTracks_DF = TAMSD_DF.loc[:, mobileTracks_List]
    # get half the track lengths
    mobileTAMSD_range = int(math.floor(mobileTAMSDTracks_DF.count().max() / 2))
    mobileTAMSD_half_indices = np.round(
        mobileTAMSDTracks_DF.index[0:mobileTAMSD_range], 3
    )
    mobileTAMSD_half_MSDs = pd.DataFrame(index=mobileTAMSD_half_indices)
    for track in mobileTAMSDTracks_DF:
        half_last_index = round(
            (mobileTAMSDTracks_DF[track].last_valid_index() / 2) / (1 / frameTime)
        ) * (1 / frameTime)
        mobileTAMSD_half_MSDs[track] = mobileTAMSDTracks_DF[track][0:half_last_index]

    # Determine Average Track of all tracks
    avgMobileTAMSD_half_MSD = mobileTAMSD_half_MSDs.mean(axis=1)
    avgStdMobileTAMSD_half_MSD = mobileTAMSD_half_MSDs.std(axis=1)

    # Plot results as half track lengths
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Plot individual tracks, set label for legend
    ax.plot(
        mobileTAMSD_half_MSDs.index,
        mobileTAMSD_half_MSDs,
        "k-",
        alpha=0.2,
        label="Mobile Tracks",
    )

    # Set the scale of the axes to 'log'
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Set the window title
    fig = plt.gcf()
    fig.canvas.set_window_title("TAMSD of Mobile Tracks")
    # Set the legend to show only one entry for tracks
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=12)
    # Set the headline/title for the plot
    fig.suptitle("TAMSD: Mobile Tracks", fontsize=20)
    # Set the axes labels
    ax.set_ylabel(r"$\overline{\delta^2 (\Delta)}$  [$\mu$m$^2$]", fontsize=15)
    ax.set_xlabel("lag times [$s$]", fontsize=15)
    # Position the axes labels
    ax.xaxis.set_label_coords(0.5, -0.07)
    # Determine the min/max values for the x, y axes
    # Padding value to increase axes by
    axes_padding = 0.1
    # Calculate min/max values for axes
    # Calculate min/max values for axes
    x_min = avgMobileTAMSD_half_MSD.index[0] - (
        avgMobileTAMSD_half_MSD.index[0] * axes_padding
    )
    x_max = avgMobileTAMSD_half_MSD.index.max() + (
        avgMobileTAMSD_half_MSD.index.max() * axes_padding
    )
    y_min = mobileTAMSD_half_MSDs.min().min() - (
        mobileTAMSD_half_MSDs.min().min() * axes_padding
    )
    y_max = mobileTAMSD_half_MSDs.max().max() + (
        mobileTAMSD_half_MSDs.max().max() * axes_padding
    )
    # Set the min/max values for the x, y axes
    ax.set(ylim=(y_min, y_max), xlim=(x_min, x_max))
    # Display the TAMSD plot
    plt.show()

    # ---------------------------------------------------------------------------
    # Plot the TAMSD Averaged Track on top of this plot, set label for legend
    # ---------------------------------------------------------------------------

    # Plot results as half track lengths
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Plot individual tracks, set label for legend
    ax.plot(
        mobileTAMSD_half_MSDs.index,
        mobileTAMSD_half_MSDs,
        "k-",
        alpha=0.2,
        label="Mobile Tracks",
    )

    ax.plot(
        avgMobileTAMSD_half_MSD.index,
        avgMobileTAMSD_half_MSD,
        "b-",
        alpha=1,
        linewidth=3,
        label="Averaged Track",
    )
    # Linear Fit of Averaged Track
    Avg_line, Avg_slope, Avg_intercept = plot_TAMSD_bestFit(
        avgMobileTAMSD_half_MSD, fit_range
    )
    ax.plot(
        Avg_line["lagt"],
        Avg_line["Avg_TAMSD"],
        "-r",
        linewidth=3,
        label="Linear Fit: y = {:.2f} x + {:.2f}".format(Avg_slope, Avg_intercept),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Set the window title
    fig = plt.gcf()
    fig.canvas.set_window_title("TAMSD of Mobile Tracks")
    # Set the legend to show only one entry for tracks
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=12)
    # Set the headline/title for the plot
    fig.suptitle("TAMSD: Mobile Tracks", fontsize=20)
    # Set the axes labels
    ax.set_ylabel(r"$\overline{\delta^2 (\Delta)}$  [$\mu$m$^2$]", fontsize=15)
    ax.set_xlabel("lag times [$s$]", fontsize=15)
    # Position the axes labels
    ax.xaxis.set_label_coords(0.5, -0.07)
    # Determine the min/max values for the x, y axes
    # Padding value to increase axes by
    axes_padding = 0.1
    # Calculate min/max values for axes
    # Calculate min/max values for axes
    x_min = avgMobileTAMSD_half_MSD.index[0] - (
        avgMobileTAMSD_half_MSD.index[0] * axes_padding
    )
    x_max = avgMobileTAMSD_half_MSD.index.max() + (
        avgMobileTAMSD_half_MSD.index.max() * axes_padding
    )
    y_min = mobileTAMSD_half_MSDs.min().min() - (
        mobileTAMSD_half_MSDs.min().min() * axes_padding
    )
    y_max = mobileTAMSD_half_MSDs.max().max() + (
        mobileTAMSD_half_MSDs.max().max() * axes_padding
    )
    # Set the min/max values for the x, y axes
    ax.set(ylim=(y_min, y_max), xlim=(x_min, x_max))
    # Display the TAMSD plot
    plt.show()

    # ---------------------------------------------------------------------------
    # Plot the TAMSD Averaged Track by itself with an error cloud
    # ---------------------------------------------------------------------------

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Plot individual tracks, set label for legend
    ax.plot(
        avgMobileTAMSD_half_MSD.index,
        avgMobileTAMSD_half_MSD,
        "b-",
        alpha=1,
        linewidth=3,
        label="Averaged Track",
    )
    plt.fill_between(
        avgMobileTAMSD_half_MSD.index,
        avgMobileTAMSD_half_MSD - avgStdMobileTAMSD_half_MSD,
        avgMobileTAMSD_half_MSD + avgStdMobileTAMSD_half_MSD,
        alpha=0.2,
    )
    ax.plot(
        Avg_line["lagt"],
        Avg_line["Avg_TAMSD"],
        "-r",
        linewidth=3,
        label="Linear Fit: y = {:.2f} x + {:.2f}".format(Avg_slope, Avg_intercept),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Set the window title
    fig = plt.gcf()
    fig.canvas.set_window_title("TAMSD of Mobile Tracks")
    # Set the legend to show only one entry for tracks
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=12)
    # Set the headline/title for the plot
    fig.suptitle("TAMSD: Mobile Tracks", fontsize=20)
    # Set the axes labels
    ax.set_ylabel(r"$\overline{\delta^2 (\Delta)}$  [$\mu$m$^2$]", fontsize=15)
    ax.set_xlabel("lag times [$s$]", fontsize=15)
    # Position the axes labels
    ax.xaxis.set_label_coords(0.5, -0.07)
    # Determine the min/max values for the x, y axes
    # Padding value to increase axes by
    axes_padding = 0.1
    # Calculate min/max values for axes
    # Calculate min/max values for axes
    x_min = avgMobileTAMSD_half_MSD.index[0] - (
        avgMobileTAMSD_half_MSD.index[0] * axes_padding
    )
    x_max = avgMobileTAMSD_half_MSD.index.max() + (
        avgMobileTAMSD_half_MSD.index.max() * axes_padding
    )
    y_min = mobileTAMSD_half_MSDs.min().min() - (
        mobileTAMSD_half_MSDs.min().min() * axes_padding
    )
    y_max = mobileTAMSD_half_MSDs.max().max() + (
        mobileTAMSD_half_MSDs.max().max() * axes_padding
    )
    # Set the min/max values for the x, y axes
    ax.set(ylim=(y_min, y_max), xlim=(x_min, x_max))
    # Display the TAMSD plot
    plt.show()

    # ---------------------------------------------------------------------------
    # Plot the EAMSD Averaged Track with fit and error cloud
    # ---------------------------------------------------------------------------
    # // TODO Step 0: Import stat_MSD_outputs.py
    # // TODO Step 1: Load the original tracks .json into memory
    # // TODO Step 2: Make a new dataframe with just the mobile tracks, same format


def genMobileEAMSDTracks(selectedTracks_DF, mobileTrack_List, savePath):
    mobileTrack_DF = pd.DataFrame()
    for particle, track in selectedTracks_DF.reset_index(drop=True).groupby("particle"):
        if particle in mobileTrack_List:
            mobileTrack_DF = mobileTrack_DF.append(track, ignore_index=True)
    return mobileTrack_DF
    # reindex by particle, if particle is in mobileList, add to a new DF, after all additions reindex and output

    # TODO Step 3: Pass new dataframe into stat_MSD_outputs.py ensaMSD func
    # TODO Step 4: Output the result as a .json of mobile_EAMSD_DF, and output mobile_EAMSD_allTracksAllLags.json as well
    # TODO Step 5: Plot the mobile_EAMSD data with an error cloud
    # TODO Step 6: Plot the mobile_EAMSD data with a fit and error cloud
    # TODO Step 7: Plot the mobile_TAMSD and mobile_EAMSD data with fits and error clouds on same plot


# ! THIS IS THE COPIED BESTFIT CODE FROM STAT_MSD
def plot_EAMSD_bestFit(msds, fit_range):
    fit_range = fit_range
    msds_vals = msds
    slope, intercept = np.polyfit(
        np.log(msds_vals["lagt"][fit_range[0] : fit_range[1]]),
        np.log(msds_vals["msd"][fit_range[0] : fit_range[1]]),
        1,
    )
    y_fit = np.exp(
        slope * np.log(msds_vals["lagt"][fit_range[0] : fit_range[1]]) + intercept
    )
    line = pd.DataFrame({"lagt": msds_vals["lagt"], "Avg_EAMSD": y_fit})
    return line, slope, intercept

    # ! YOU MAY HAVE TO REDO THE ENSEMBLE CALC IF YOU ARE USING THE OLD DATAFRAME BECAUSE IT
    # ! INCLUDES ALL POINTS
    # ! REVISIT HOW YOU DID THE ERROR CLOUD FOR EAMSD< IT IS LIKELY WRONG
    # ! Step 1: recalc ensemble MSD for mobile tracks only
    # ! STEP 1: MAY NO HAVE TO RECALC IF YOU HAVE THE AGGREGATED OUTPUT !!! WOO :D:D:D:D
    # ! Step 2: Output that as a .json file
    # ! Step 3: Then plot the corresponding data with a fit and error cloud

    # ! THIS IS THE FUNCTION I AM CURRENTLY WRITING


def plot_AvgMobileEAMSD(EAMSD_DF, mobileTracks_List, frameTime, fit_range):
    return None

    # ! BELOW THIS IS THE COPIED CODE FROM STAT_MSD


def plot_EAMSD(ensa_msds, fit_range):
    ensa_msds = ensa_msds
    fit_range = fit_range
    # Plot results as half the track lengths by modifiying plotting window
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot EAMSD of tracks
    ax.plot(ensa_msds["lagt"], ensa_msds["msd"], "o", label="Ensemble Average MSD")
    # Determine linear fit to data
    # Set number of initial points to fit
    # Fit from EAMSD calcs
    line, slope, intercept = plot_EAMSD_bestFit(ensa_msds, fit_range)
    # Plot linear fit of EAMSD data
    ax.plot(
        line["lagt"],
        line["Avg_EAMSD"],
        "-r",
        linewidth=3,
        label="Linear Fit: y = {:.2f} x + {:.2f}".format(slope, intercept),
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
    ax.set_ylabel(r"$\langle$$\bf{r}$$^2$($\Delta)\rangle$ [$\mu$m$^2$]", fontsize=15)
    ax.set_xlabel("lag time [$s$]", fontsize=15)
    # Position the axes labels
    ax.xaxis.set_label_coords(0.5, -0.07)
    # Determine the min/max values for the x, y axes
    # Padding value to increase axes by
    axes_padding = 0.1
    # Calculate min/max values for axes
    x_min = ensa_msds["lagt"].min() - (ensa_msds["lagt"].min() * axes_padding)
    x_max = ensa_msds["lagt"].max() + (ensa_msds["lagt"].max() * axes_padding)
    y_min = ensa_msds["msd"].min() - (ensa_msds["msd"].min() * axes_padding)
    y_max = ensa_msds["msd"].max() + (ensa_msds["msd"].max() * axes_padding)
    # Set the min/max values for x, y axes
    ax.set(ylim=(y_min, y_max), xlim=(x_min, x_max))
    # Display the EAMSD plot
    plt.show()


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
    # Path to load selected_track_list.json from track_selector.py
    jsonTracksLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"

    # Paths to MSD .json files to load as dataframes
    jsonSelectedTracksLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"
    jsonTAMSDLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/TAMSD.json"
    jsonEAMSDLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/EAMSD.json"
    # Path to load Mobile Trapped Tracks dict
    jsonMobileTrappedDictPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/Mobile_Trapped_tracks.json"

    # TODO Disabled for now
    # jsonAllDisplacementsLoadPath = r"/home/vivek/Documents/Piezo1/temp_outputs/Statistics/MSDs/All_lag_displacements_microns.json"

    # Path to main directory for saving outputs
    savePath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs"

    # Experimental parameters
    pixelWidth = 0.1092  # in microns
    frameTime = 100  # in milliseconds, typical value is 50 or 100

    # Local Error (determines mobile vs. trapped tracks) in um^2
    # Calculation of localization error is normally distributed with a stdev 12.7nm
    # or (0.117 pixels) in each direction
    # Sqrt[2 * (12.7 ** 2) ] = 0.018
    # Try defining mobile tracks as those with taMSD(1 sec) > 0.018 um^2
    localError = 0.018

    # The lag time at which to check this cutoff (in seconds)
    # Can use any value compatible with frameTime (any multiple of frameTime)
    localErrorLagTime = 1.0

    # Range of data to fit to a line
    fit_range = [1, 30]  # bounding indices for tracks to fit, select linear region
    # * -----END OF USER INPUTS----- * #

    # * -----START SUBROUTINE----- * #
    # Convert frameTime to frames-per-second
    frameTime = 1000 / frameTime

    # Instantiate the json_converter class
    jc = json_converter()

    # Load tracks data into a pandas DataFrame
    selectedTracks_DF = jc.json_SelectedTracks_to_DF(jsonSelectedTracksLoadPath)
    TAMSD_DF = pd.read_json(jsonTAMSDLoadPath, orient="split")
    EAMSD_DF = pd.read_json(jsonEAMSDLoadPath, orient="split")
    # * Disabled for now, will enable if use-case arises
    # AllDisplacements_DF = pd.read_json(jsonAllDisplacementsLoadPath, orient="split")

    # Setup the index for TAMSD
    TAMSD_DF.set_index("lagt", inplace=True)

    # Define mobile and trapped tracks
    mobileTrappedTracks_Dict = mobile_trapped_tracks(
        TAMSD_DF, localError, localErrorLagTime, savePath
    )

    # Load mobile and trapped tracks data from previously generated .json output
    # ! Uncomment below to use the loader
    # testLoadMobileTrappedTracks_Dict = load_MobileTrapped_json(jsonMobileTrappedDictPath)

    # Plot the Trapped Tracks without any fitting
    plot_TrappedTAMSD(TAMSD_DF, mobileTrappedTracks_Dict["Trapped"], frameTime)

    # Plot the TAMSD Mobile Tracks without any fitting
    # Then...
    # Plot the TAMSD Average Track on top of the Mobile Tracks
    # Then...
    # Plot the TAMSD Average Track with Error and Fitting
    plot_MobileTAMSD(TAMSD_DF, mobileTrappedTracks_Dict["Mobile"], frameTime, fit_range)

    # Take the selected track list, and produce a new DF with just the mobile tracks
    mobileTracks_DF = genMobileEAMSDTracks(
        selectedTracks_DF, mobileTrappedTracks_Dict["Mobile"], savePath
    )

    # * -----END   SUBROUTINE----- * #

    # ! -----DEBUGGING CODE START----- ! #

    # Plot EAMSD of Mobile Tracks with Fit
    plot_AvgMobileEAMSD(
        EAMSD_DF, mobileTrappedTracks_Dict["Mobile"], frameTime, fit_range
    )

    # Plot both AVG TAMSD AND AVG EAMSD on same plot, both with ERROR CLOUDS
    # Then...
    # Plot both AVGs with both FITS on same plot
    def plot_AvgMobileTA_EA_MSD(
        TAMSD_DF, EAMSD_DF, mobileTracks_List, frameTime, fit_range
    ):
        pass


# ! After this point you will begin histogramming alpha values from the TAMSD ...

# ! -----DEBUGGING CODE   END----- ! #


# --------------------------------------------------------------------------------------------------
# // TODO 1.0   Load a .json of all tracks
# // TODO 2.0   Use selection criterion to differentiate mobile and trapped tracks
# TODO 3.0   Plot Mobile and Trapped Tracks with output to file option
# TODO 4.0   TAMSD of mobile tracks
# TODO 4.1   --Power Law fit
# TODO 4.2   --Linear Fit
# TODO 5.0   EAMSD of mobile tracks
# TODO 5.1   --Power Law fit
# TODO 5.2   --Linear Fit
# TODO 6.0   TAMSD and EAMSD mobile tracks, both on same plot
# TODO 6.1   --Linear Fit
# TODO 6.2   --Linear Fit on Log Data
# TODO 7.0   Plot any individual mobile or trapped track to visualize its path
# TODO 8.0   Distribution of alpha values TAMSD for upto 10% trajectory time
