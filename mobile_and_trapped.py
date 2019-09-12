import codecs
import json
import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


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
        print(trappedTAMSDTracks_DF[track][0.3])
        print(trappedTAMSD_half_MSDs[track][0.3])

        # trappedTAMSD_half_MSDs = trappedTAMSD_half_MSDs.join(trappedTAMSDTracks_DF[track][0:half_last_index])
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


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
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
    # Can use any value compatible with frameTime
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

    # Plot the trapped and mobile tracks on separate plots without any fitting
    plot_TrappedTAMSD(TAMSD_DF, mobileTrappedTracks_Dict["Trapped"], frameTime)
    # * -----END   SUBROUTINE----- * #

    # ! -----DEBUGGING CODE START----- ! #

    def plot_MobileTAMSD(TAMSD_DF, mobileTracks_List):
        pass

    def plot_AvgMobileTAMSD(TAMSD_DF, mobileTracks_List):
        pass

    def plot_AvgMobileTAMSD_PowerFit(TAMSD_DF, mobileTracks_List, fitRange):
        pass

    def plot_AvgMobileTAMSD_LinearFit(TAMSD_DF, mobileTracks_List, fitRange):
        pass

    def plot_AvgMobileEAMSD(EAMSD_DF, mobileTracks_List):
        pass

    # Plot both of them on the same plot
    def plot_AvgMobileTA_EA_MSD(TAMSD_DF, EAMSD_DF, mobileTracks_List):
        pass

    def plot_AvgMobileEAMSD_PowerFit(EAMSD_DF, mobileTracks_List, fitRange):
        pass

    def plot_AvgMobileEAMSD_LinearFit(EAMSD_DF, mobileTracks_List, fitRange):
        pass


# ! After this point you will begin histogramming alpha values from the TAMSD ...

# ! -----DEBUGGING CODE   END----- ! #


# --------------------------------------------------------------------------------------------------
# // TODO 1.0   Load a .json of all tracks
# // TODO 2.0   Use selection criterion to differentiate mobile and trapped tracks
# TODO 3.0   Plot Mobile and Trapped Tracks with output to file option
# TODO 4.0   TAMSD of mobile tracks
# TODO 4.1   --Power Law fit
# TODO 4.2   --Slope Fit
# TODO 5.0   EAMSD of mobile tracks
# TODO 5.1   --Power Law fit
# TODO 5.2   --Slope Fit
# TODO 6.0   TAMSD and EAMSD mobile tracks, both on same plot
# TODO 6.1   --Linear Fit
# TODO 6.2   --Linear Fit on Log Data
# TODO 7.0   Plot any individual mobile or trapped track to visualize its path
# TODO 8.0   Distribution of alpha values TAMSD for upto 10% trajectory time
