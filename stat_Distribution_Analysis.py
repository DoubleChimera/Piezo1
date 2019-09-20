import codecs
import json
import math

# import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stat_MSD_outputs as statMSDo


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

    def load_MobileTrapped_json(self, jsonFilePath):
        with open(jsonFilePath) as json_file:
            jdata = json.load(json_file)
        return jdata


def loadMobileTracksDF(selectedTracks_DF, mobileTrack_List, savePath):
    mobileTracks_DF = pd.DataFrame()
    for particle, track in selectedTracks_DF.reset_index(drop=True).groupby("particle"):
        if particle in mobileTrack_List:
            mobileTracks_DF = mobileTracks_DF.append(track, ignore_index=True)
    return mobileTracks_DF


def alphaValAnalysis(TAMSD_DF, mobileTracks_List, cutoffPercentLength, binWidth=0.05):
    # Use the mobileTracks_List to make a DF of mobile TAMSD
    mobileTAMSDTracks_DF = TAMSD_DF.loc[:, mobileTracks_List]
    # Insert the 'lagt' column into the mobileTAMSD DF
    mobileTAMSDTracks_DF.insert(0, "lagt", TAMSD_DF["lagt"], True)
    # Setup a new DF to store the alpha values into
    alphaVals_DF = pd.DataFrame(columns=["particle", "alpha_vals"])
    alphaVals_DF["particle"] = list(mobileTAMSDTracks_DF.set_index("lagt"))
    # Re-index alphaVals_DF to particle in prep for iteration
    alphaVals_DF.set_index("particle", inplace=True)
    # Go over all particles/tracks and determine alpha vals, insert into alphaVals_DF
    for particle in mobileTAMSDTracks_DF.set_index("lagt"):
        # Determine cutoffPercentLength of track
        cutoffLength = int(
            math.floor(
                mobileTAMSDTracks_DF[particle].count() * (cutoffPercentLength / 100)
            )
        )
        # Perform a linear fit for only these points
        cutoffSlope, cutoffIntercept = np.polyfit(
            np.log(mobileTAMSDTracks_DF["lagt"].iloc[0:cutoffLength]),
            np.log(mobileTAMSDTracks_DF[particle].iloc[0:cutoffLength]),
            1,
        )
        alphaVals_DF.loc[particle] = cutoffSlope
    # Calc the mean alpha val and print it to console
    meanAlphaVal = alphaVals_DF.mean(axis=0)
    print(f"Mean {meanAlphaVal}")
    # Make a histogram of probabilities out of the alpha values
    # --Setup the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # --Setup the bin width ranges
    alphaMaxBin = math.ceil(alphaVals_DF["alpha_vals"].max() * 20) / 20
    alphaHistBinRange = np.arange(0, alphaMaxBin + binWidth, binWidth)
    # --Draw the plot
    ax.hist(
        list(alphaVals_DF["alpha_vals"]),
        bins=alphaHistBinRange,
        color="gray",
        alpha=0.7,
        edgecolor="black",
    )
    # --Title and labels
    ax.set_title("Alpha Value Distributions")
    ax.set_xlabel("Alpha Values")
    ax.set_ylabel("Probability")
    # --Show the plot
    plt.show()


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
    # Paths to MSD .json files to load as dataframes
    # Selected Tracks DF
    jsonSelectedTracksLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"
    # TAMSD of ALL Tracks DF (Trapped and Mobile)
    jsonTAMSDLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/TAMSD.json"
    # Dict -List of Mobile and Trapped Tracks
    jsonMobileTrappedDictPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/Mobile_Trapped_tracks.json"
    # ALL Tracks ALL Lags DF (Trapped and Mobile)
    jsonAllTracksAllLags = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/All_Lagtimes.json"

    # Path to main directory for saving outputs
    savePath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs"

    # Experimental parameters
    pixelWidth = 0.1092  # in microns
    frameTime = 100  # in milliseconds, typical value is 50 or 100

    # Local Error (determines mobile vs. trapped tracks) in um^2
    # Calculation of localization error is normally distributed with a stdev 12.7nm
    # or (0.127 pixels) in each direction
    # Sqrt[2 * (12.7 ** 2) ] = 0.018
    # Try defining mobile tracks as those with taMSD(1 sec) > 0.018 um^2
    localError = 0.018
    # The lag time at which to check this cutoff (in seconds)
    # Can use any value compatible with frameTime (any multiple of frameTime)
    localErrorLagTime = 1.0

    # Range of data to fit to a line
    fit_range = [1, 30]  # bounding indices for tracks to fit, select linear region
    # * -----END OF USER INPUTS----- * #

    # * ----- START SUBROUTINE ----- * #
    # Convert frameTime to frames-per-second
    frameTime = 1000 / frameTime

    # Instantiate the json_converter class
    jc = json_converter()

    # Load tracks data into a pandas DataFrame
    selectedTracks_DF = jc.json_SelectedTracks_to_DF(jsonSelectedTracksLoadPath)
    TAMSD_DF = pd.read_json(jsonTAMSDLoadPath, orient="split")
    AllTracksAllLags_DF = pd.read_json(jsonAllTracksAllLags, orient="split")
    MobileTrappedTracks_Dict = jc.load_MobileTrapped_json(jsonMobileTrappedDictPath)

    # Histogram the alpha values for the TAMSD of Mobile Tracks up to a cutoff percentage
    alphaValAnalysis(TAMSD_DF, MobileTrappedTracks_Dict["Mobile"], 10)

    # * -----  END SUBROUTINE  ----- * #

    # ! ----- START DEBUGGING  ----- ! #
    # Cumulative Distributon Function, lagtime_limit is in number of frames and inclusive
    def cumulDistrib(
        AllTracksAllLags_DF,
        MobileTracks_List,
        frameTime,
        lagtime_limit=10,
        outputPlotRange=1,
    ):
        # Renames the index of AllTracksAllLags_DF to 'particle'
        AllTracksAllLags_DF.index.names = ["particle"]
        # Gen mobileTracksAllLags_DF upto the lagtime_limit var
        mobileTracksAllLags_DF = AllTracksAllLags_DF.loc[
            MobileTracks_List, "lagt":f"y_lag{lagtime_limit}"
        ]
        # Convert 'lagt' column to frames by multiplying by 10 and casting as int
        mobileTracksAllLags_DF.loc[:, "lagt"] *= 10
        mobileTracksAllLags_DF.lagt = mobileTracksAllLags_DF.lagt.astype(int)
        # Rename 'lagt' column to 'frame'
        mobileTracksAllLags_DF.rename(columns={"lagt": "frame"}, inplace=True)
        # Set 'particle' and 'frame' as the indices for mobileTracksAllLags_DF
        mobileTracksAllLags_DF.reset_index(inplace=True)
        mobileTracksAllLags_DF.set_index(["particle", "frame"], inplace=True)
        # make a list of all unique indices corresponding to particle numbers, use that list as the index for this DF.
        particle_List = mobileTracksAllLags_DF.reset_index().particle.unique()
        # Gen an empty DF for r^2 and CDF results
        cumulDistrib_DF = pd.DataFrame(particle_List, columns=["particle"]).set_index(
            "particle"
        )
        # Setup the prefix and suffix for the column names
        CDF_columns_prefix = ["r2", "CDF"]
        CDF_columns_suffix = ["_lag{}".format(l) for l in range(lagtime_limit + 1)]
        # Iterate over the prefixes and suffixes to produce the column names list
        stat_MSD = statMSDo.stat_MSD()
        CDF_columns_results = list(
            stat_MSD.genLagColumns(CDF_columns_suffix, CDF_columns_prefix)
        )
        # Delete the useless 'CDF_lag0' column name
        CDF_columns_results.remove("CDF_lag0")
        # Append the columns names list to the cumulDistrib_DF
        cumulDistrib_DF = cumulDistrib_DF.reindex(
            columns=cumulDistrib_DF.columns.tolist() + CDF_columns_results
        )

        for particle in particle_List:
            # Calculate and insert all r^2 values into cumulDistrib_DF
            for lag in range(lagtime_limit + 1):
                # for every lag, find the r^2,
                xy_vals = np.array(
                    mobileTracksAllLags_DF.loc[
                        (particle, 0), [f"x_lag{lag}", f"y_lag{lag}"]
                    ]
                )
                r2_temp = (xy_vals[0] ** 2) + (xy_vals[1] ** 2)
                # Store result in cumulDistrib_DF in r2_lag{lag}
                cumulDistrib_DF.at[particle, f"r2_lag{lag}"] = r2_temp
        # At this point all the cumulDistrib lags have been inserted
        # Now calc all cumulative distribution function values
        # Loop over particle list
        for particle in particle_List:
            # Loop over each lag in particle list
            for lag in np.arange(lagtime_limit) + 1:
                # Determine total number of tracks, for later division
                CDF_total_tracks = cumulDistrib_DF[f"r2_lag{lag}"].count()
                # Extract value to compare against the rest of the list of values
                r2_to_compare = float(cumulDistrib_DF.loc[particle, [f"r2_lag{lag}"]])
                # Determine how many values are less than or equal to it
                CDF_temp = (cumulDistrib_DF[f"r2_lag{lag}"] <= r2_to_compare).sum()
                # Divide that total by the number of tracks
                CDF_temp = CDF_temp / CDF_total_tracks
                # Insert that value in the appropriate place in the cumulDistrib_DF
                cumulDistrib_DF.at[particle, f"CDF_lag{lag}"] = CDF_temp
        # Plot the cumulative distribution function
        for plotNum in np.arange(outputPlotRange) + 1:
            # Grab the columns of interest
            plotCDF_DF = cumulDistrib_DF[
                [f"r2_lag{plotNum}", f"CDF_lag{plotNum}"]
            ].copy()
            # Sort by the r2 column
            plotCDF_DF = plotCDF_DF.sort_values(
                [f"r2_lag{plotNum}", f"CDF_lag{plotNum}"], ascending=[True, False]
            )
            print(plotCDF_DF)
            # Plot the two columns with r2 as 'x' and CDF as 'y'
            plt.step(plotCDF_DF[f"r2_lag{plotNum}"], plotCDF_DF[f"CDF_lag{plotNum}"])
            # ! Need to pretty up this plot with nice formatting
            plt.show()

    cumulDistrib(AllTracksAllLags_DF, MobileTrappedTracks_Dict["Mobile"], frameTime)
    # ! -----  END DEBUGGING   ----- ! #

    # * ----- SCRAP CODE BELOW FOR REFERENCE----- * #
"""
    # Use the mobileTracks_List to make a DF of mobile TAMSD
    mobileTAMSDTracks_DF = TAMSD_DF.loc[:, MobileTrappedTracks_Dict["Mobile"]]

    # Insert the 'lagt' column into the mobileTAMSD DF
    mobileTAMSDTracks_DF.insert(0, "lagt", TAMSD_DF["lagt"], True)

    # Print the mobile TAMSD DF
    print(mobileTAMSDTracks_DF)
"""
"""
    # This accesses the 0th particle, with all tracks and all lags
    # print(AllTracksLags_DF.loc[0])

    # This accesses a specific column, such as x_lag1 or y_lag1 of that data
    # particularly, this slices all the data in the represented range
    # print(AllTracksLags_DF.loc[0, 'x_lag0':'y_lag2'])

    # Use the mobileTracks_List to make a DF of mobile TAMSD
    # mobileTAMSDTracks_DF = TAMSD_DF.loc[:, MobileTrappedTracks_Dict["Mobile"]]

    # Insert the 'lagt' column into the mobileTAMSD DF
    # mobileTAMSDTracks_DF.insert(0, "lagt", TAMSD_DF["lagt"], True)

    # Print the mobile TAMSD DF with 'lagt' as an index
    # print(mobileTAMSDTracks_DF.set_index('lagt'))

    # for column in mobileTAMSDTracks_DF.set_index('lagt'):
    #     print(column)

"""
