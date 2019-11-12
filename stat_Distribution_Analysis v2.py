import codecs
import json
import math

# import os.path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np

# from lmfit.model import Model
import pandas as pd
import stat_MSD_outputs as statMSDo
from scipy.optimize import curve_fit
from scipy import stats


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
    # First show the alpha values of All Tracks
    # Make a copy of the TAMSD_DF
    AllTAMSDTracks_DF = TAMSD_DF.copy(deep=True)
    # Insert a '0' MSD value for lagtime 0 at the beginning of the DF
    # --Make an empty list as long as the particle count
    zeroLagList = np.empty((len(AllTAMSDTracks_DF.columns),), dtype=np.int)
    # --Fill the empty list with zeroes
    zeroLagList.fill(0)
    # --Insert that list at the -1 indexed row
    AllTAMSDTracks_DF.loc[-1] = zeroLagList
    # --Increase all indexes by 1, making the -1 index zero
    AllTAMSDTracks_DF.index = AllTAMSDTracks_DF.index + 1
    # --Sort the DF by index values
    AllTAMSDTracks_DF = AllTAMSDTracks_DF.sort_index()
    # Setup a new DF to store the alpha values into
    alphaValsAll_DF = pd.DataFrame(columns=["particle", "alpha_vals"])
    alphaValsAll_DF["particle"] = list(AllTAMSDTracks_DF.set_index("lagt"))
    # Re-index alphaValsAll_DF to particle in prep for iteration
    alphaValsAll_DF.set_index("particle", inplace=True)
    # Go over all particles/tracks and determine alpha vals, insert into alphaValsAll_DF
    for particle in AllTAMSDTracks_DF.set_index("lagt"):
        # Determine cutoffPercentLength of track
        cutoffIndex = int(
            math.ceil(
                (math.floor(AllTAMSDTracks_DF[particle].count() / 2) + 1)
                * (cutoffPercentLength / 100)
            )
        )
        cutoffSlope, cutoffIntercept, cutoffR_value, cutoffP_value, cutoffStdErr = stats.linregress(
            np.log(AllTAMSDTracks_DF["lagt"].iloc[1:cutoffIndex]),
            np.log(AllTAMSDTracks_DF[particle].iloc[1:cutoffIndex]),
        )
        alphaValsAll_DF.loc[particle] = cutoffSlope
        # print(f'a-value {particle} : {cutoffSlope}')
        # print(f'r-squared val      : {cutoffR_value**2}')
    # Calc the mean alpha val and print it to console
    meanAlphaVal = alphaValsAll_DF.mean(axis=0)
    print(f"Mean All: {meanAlphaVal}")
    # Make a histogram of probabilities out of the alpha values
    # --Setup the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # --Setup the bin width ranges
    alphaMaxBin = math.ceil(alphaValsAll_DF["alpha_vals"].max() * 20) / 20
    alphaHistBinRange = np.arange(-0.2, alphaMaxBin + binWidth, binWidth)
    weights = np.ones_like(alphaValsAll_DF["alpha_vals"]) / float(
        len(alphaValsAll_DF["alpha_vals"])
    )
    # --Draw the plot
    ax.hist(
        list(alphaValsAll_DF["alpha_vals"]),
        bins=alphaHistBinRange,
        weights=weights,
        color="gray",
        alpha=0.7,
        edgecolor="black",
    )
    # --Set the xlim
    plt.xlim(xmin=-0.2, xmax=2.0)
    # --Title and labels
    ax.set_title("Alpha Value Distributions of All Tracks")
    ax.set_xlabel("Alpha Values")
    ax.set_ylabel("Probability")
    # --Show the plot
    plt.show()

    # Calc and plot the alpha values of the mobile tracks
    # Use the mobileTracks_List to make a DF of mobile TAMSD
    mobileTAMSDTracks_DF = TAMSD_DF.loc[:, mobileTracks_List]
    # Insert the 'lagt' column into the mobileTAMSD DF
    mobileTAMSDTracks_DF.insert(0, "lagt", TAMSD_DF["lagt"], True)
    # Setup a new DF to store the alpha values into
    alphaValsMobile_DF = pd.DataFrame(columns=["particle", "alpha_vals"])
    alphaValsMobile_DF["particle"] = list(mobileTAMSDTracks_DF.set_index("lagt"))
    # Re-index alphaValsMobile_DF to particle in prep for iteration
    alphaValsMobile_DF.set_index("particle", inplace=True)
    # Go over all particles/tracks and determine alpha vals, insert into alphaValsMobile_DF
    for particle in mobileTAMSDTracks_DF.set_index("lagt"):
        # Determine cutoffPercentLength of track
        cutoffIndex = int(
            math.floor(
                mobileTAMSDTracks_DF[particle].count() * (cutoffPercentLength / 100)
            )
        )
        # Perform a linear fit for only these points
        cutoffSlope, cutoffIntercept = np.polyfit(
            np.log(mobileTAMSDTracks_DF["lagt"].iloc[0:cutoffIndex]),
            np.log(mobileTAMSDTracks_DF[particle].iloc[0:cutoffIndex]),
            1,
        )
        alphaValsMobile_DF.loc[particle] = cutoffSlope
    # Calc the mean alpha val and print it to console
    meanAlphaVal = alphaValsMobile_DF.mean(axis=0)
    print(f"Mean Mobile: {meanAlphaVal}")
    # Make a histogram of probabilities out of the alpha values
    # --Setup the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # --Setup the bin width ranges
    alphaMaxBin = math.ceil(alphaValsMobile_DF["alpha_vals"].max() * 20) / 20
    alphaHistBinRange = np.arange(0, alphaMaxBin + binWidth, binWidth)
    weights = np.ones_like(alphaValsMobile_DF["alpha_vals"]) / float(
        len(alphaValsMobile_DF["alpha_vals"])
    )
    # --Draw the plot
    ax.hist(
        list(alphaValsMobile_DF["alpha_vals"]),
        bins=alphaHistBinRange,
        weights=weights,
        range=[0, 2.0],
        color="gray",
        alpha=0.7,
        edgecolor="black",
    )
    # --Set the axes limits
    # --Title and labels
    ax.set_title("Alpha Value Distributions of Mobile Tracks")
    ax.set_xlabel("Alpha Values")
    ax.set_ylabel("Probability")
    # --Show the plot
    plt.show()


# Cumulative Distributon Function, lagtime_limit is in units of number of frames and inclusive
# Single Exponential Fitting Equation
def func_cdfONEmob(x, d):
    return 1 - np.exp(-(x / d))


# Double Exponential Fitting Equation
def func_cdfTWOmob(x, w, d_1, d_2):
    return 1 - ((w * np.exp(-x / d_1)) + ((1 - w) * np.exp(-x / d_2)))


# Random walk on a fractal (Random Fractal Modeled by Percolation Cluster)
def func_cdfRandFract(x, d_f, dim, d_w, u, t):
    return (x ** (d_f - dim) / t ** (d_f / d_w)) * np.exp(-(x / (t ** (1 / d_w))) ** u)


def cumulDistrib(
    AllTracksAllLags_DF,
    MobileTracks_List,
    frameTime,
    lagtime_limit=10,
    outputPlotLagRange=10,
):
    # Check that lagtime limit is less than or equal to the available lags in AllTracksAllLags_DF
    # --Determine number of lags stored in AllTracksAllLags_DF
    storedLags = int(((AllTracksAllLags_DF.shape[1] - 6) / 2))
    # --Check if lagtime_limit is greater than the stored number of lags
    if lagtime_limit > storedLags:
        # If it is, reduce the lagtime_limit to the max number of lags available
        print("NOTE: lagtime_limit greater than stored number of lags...")
        print(f"Adjusting lagtime limit from: {lagtime_limit}  to: {storedLags}")
        lagtime_limit = storedLags
    # Check that the ouputPlotLagRange is less than or equal to the adjusted lagtime_limit
    if outputPlotLagRange > lagtime_limit:
        outputPlotLagRange = lagtime_limit
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
    # ! Unused for now, but here incase we need it
    # particle_List = mobileTracksAllLags_DF.reset_index().particle.unique()
    CDF_columns_prefix = ["r2", "CDF"]
    CDF_columns_suffix = ["_lag{}".format(l) for l in range(1, lagtime_limit + 1)]
    # Iterate over the prefixes and suffixes to produce the column names list
    stat_MSD = statMSDo.stat_MSD()
    CDF_columns_results = list(
        stat_MSD.genLagColumns(CDF_columns_suffix, CDF_columns_prefix)
    )
    # Generate an empty DataFrame with the columns names list to store values into
    cumulDistrib_DF = pd.DataFrame(columns=CDF_columns_results)
    # Loop over all the lagtimes
    for lag in range(1, lagtime_limit + 1):
        # Grab the x,y for all particles at a lag time and square the values
        squaredLagXY = np.square(
            AllTracksAllLags_DF.loc[:, f"x_lag{lag}":f"y_lag{lag}"]
        )
        # Insert the values into r2_lag{lag} column in the cumulDistrib_DF
        cumulDistrib_DF.loc[:, f"r2_lag{lag}"] = (
            squaredLagXY.loc[:, f"x_lag{lag}"]
            .add(squaredLagXY.loc[:, f"y_lag{lag}"])
            .reset_index(drop=True)
        )
        # Replace any float(0) values in that column with NaN
        cumulDistrib_DF.loc[:, f"r2_lag{lag}"].replace(float(0), np.nan, inplace=True)
        # Sort the values in that column in ascending order and drop NaNs
        CDF_sortedTemp = cumulDistrib_DF.sort_values(f"r2_lag{lag}")
        # Reset the index and replace the designated column
        cumulDistrib_DF.loc[:, f"r2_lag{lag}"] = CDF_sortedTemp.loc[
            :, f"r2_lag{lag}"
        ].reset_index(drop=True)
        # Determine the number of relevant (non NaN) points in that column
        CDF_pointCount = cumulDistrib_DF.loc[:, f"r2_lag{lag}"].count()
        # Make an empty list to store CDF values into
        CDF_valueList = []
        # Loop over each point in the column that is not NaN, dont use index, it will unsort
        for r2_IndexValue in range(CDF_pointCount):
            # Store each lag point as a variable
            r2_to_compare = float(
                cumulDistrib_DF.loc[:, f"r2_lag{lag}"][[r2_IndexValue]]
            )
            # Determine the index of the last valid entry (non NaN)
            CDF_lastValidIndex = cumulDistrib_DF.loc[
                :, f"r2_lag{lag}"
            ].last_valid_index()
            # Extract the column of data we are interested in for the current CDF calc up to the last non NaN index
            CDF_tempColumn = cumulDistrib_DF.loc[:CDF_lastValidIndex, f"r2_lag{lag}"]
            # Comparing to the relevant column, determine how many are less than or equal to it
            CDF_numLess = (np.array(CDF_tempColumn) <= r2_to_compare).sum()
            # Divide CDF_numLess by the total number of relevant points in that column
            CDF_value = CDF_numLess / CDF_pointCount
            # Append the value to the CDF_valueList
            CDF_valueList.append(CDF_value)
        # Insert that value in the appropriate column in the cumulDistrib_DF
        cumulDistrib_DF.loc[:, f"CDF_lag{lag}"] = pd.Series(CDF_valueList)
    # After the DF is generated, drop rows with all NaN values
    cumulDistrib_DF.dropna(axis=0, how="all", inplace=True)
    print(cumulDistrib_DF)

    # ! Here goes the code to bin CDF values

    # Setup a new empty dataframe for wValsMob_DF with the adjusted outputPlotLagRange as lagtimes in the index
    # Generate index of wValsMob_DF, will have to divide this by frameTime later
    wValsMob_index = np.linspace(
        1, outputPlotLagRange, outputPlotLagRange, endpoint=True
    )
    # Generate the new wValsMob_DF
    wValsMob_DF = pd.DataFrame(columns=["lagt", "wVal", "wVal_stdErr"])
    # Insert the index values that were previously calculated
    wValsMob_DF["lagt"] = wValsMob_index
    # Divide the index by frameTime
    wValsMob_DF.loc[:, "lagt"] /= frameTime
    # Set the 'lagt' column as the index
    wValsMob_DF.set_index("lagt", inplace=True)

    # TODO Plotter subroutine to plot the first lag and output the other plots to a save folder
    # Plot the cumulative distribution function
    for plotNum in range(1, outputPlotLagRange + 1):
        # Grab the columns of with the current lagtime
        plotCDF_DF = cumulDistrib_DF[[f"r2_lag{plotNum}", f"CDF_lag{plotNum}"]].copy()
        # Sort by the r2 column
        plotCDF_DF = plotCDF_DF.sort_values(
            [f"r2_lag{plotNum}", f"CDF_lag{plotNum}"], ascending=[True, False]
        )
        # Remove any trailing NaNs from the DF
        plotCDF_DF.dropna(axis=0, how="all", inplace=True)
        # Plot the two columns with r2 as 'x' and CDF as 'y'
        # Gather the x y data
        CDF_x_data = plotCDF_DF[f"r2_lag{plotNum}"]
        CDF_y_data = plotCDF_DF[f"CDF_lag{plotNum}"]
        # ! TEST_1 Start --A possibly better way to get the fit and residuals?
        # ! Left here for future reference
        # TEST_1_modCDFoneMOB = Model(func_cdfONEmob)
        # TEST_1_modCDFtwoMOB = Model(func_cdfTWOmob)
        # TEST_1_modONEparams = TEST_1_modCDFoneMOB.make_params(d=0.5)
        # TEST_1_modTWOparams = TEST_1_modCDFtwoMOB.make_params(w=0.5, d_1=0.01, d_2=0.1)
        # TEST_1_modONEresults = TEST_1_modCDFoneMOB.fit(CDF_y_data, TEST_1_modONEparams, x=CDF_x_data)
        # TEST_1_modTWOresults = TEST_1_modCDFtwoMOB.fit(CDF_y_data, TEST_1_modTWOparams, x=CDF_x_data)
        # print(TEST_1_modONEresults.fit_report())
        # print(TEST_1_modTWOresults.fit_report())
        # ! TEST_1 End
        # Generate fitted curve and residuals for cdfONEmob
        CDF1_popt, CDF1_pcov = curve_fit(func_cdfONEmob, CDF_x_data, CDF_y_data)
        CDF1_residuals = CDF_y_data - func_cdfONEmob(CDF_x_data, CDF1_popt)
        # Generate fitted curve and residuals for cdfTWOmob
        CDF2_popt, CDF2_pcov = curve_fit(
            func_cdfTWOmob, CDF_x_data, CDF_y_data, p0=[0.5, 0.01, 0.1]
        )
        # Weighting Factor determination
        # Determine the stdDev for the wValue perr = np.sqrt(np.diag(pcov))
        CDF2_popt_stdErr = np.sqrt(np.diag(CDF2_pcov))
        # Determine the current lagtime being evaluated
        wVal_lag = round(plotNum / frameTime, 1)
        # Evaluate the w value based on d_2 > d_1 adjust accordingly
        if CDF2_popt[2] < CDF2_popt[1]:
            # If d_2 < d_1, use 1-wVal instead of wVal
            wValsMob_DF.loc[wVal_lag, "wVal"] = 1 - CDF2_popt[0]
        else:
            # if d_2 > d_1, use wVal
            wValsMob_DF.loc[wVal_lag, "wVal"] = CDF2_popt[0]
        wValsMob_DF.loc[wVal_lag, "wVal"] = CDF2_popt[0]
        # Insert the stdDev value for the corresponding wVal
        wValsMob_DF.loc[wVal_lag, "wVal_stdErr"] = CDF2_popt_stdErr[0]
        CDF2_residuals = CDF_y_data - func_cdfTWOmob(CDF_x_data, *CDF2_popt)

        # Setup the new figure for CDF plots
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(6, 9), gridspec_kw={"height_ratios": [4, 8]}, sharex=True
        )
        # Plot original data
        ax1.plot(CDF_x_data, CDF_y_data, label="Data", color="black")
        # Plot the curve for cdfONEmob
        ax1.step(
            CDF_x_data,
            func_cdfONEmob(CDF_x_data, *CDF1_popt),
            color="C0",
            label="Single",
            linestyle="dashed",
        )
        # Plot the curve for cdfTWOmob
        ax1.step(
            CDF_x_data,
            func_cdfTWOmob(CDF_x_data, *CDF2_popt),
            color="C1",
            label="Double",
            linestyle="dashed",
        )
        # Determine current lagTime rounded to 2 decimal places
        currLagTime = round(plotNum / frameTime, 2)
        # Set the label for the x-axis and y-axis
        ax1.set_xlabel(r"$r^2$ ($\mu$m$^2$)", fontsize=15)
        ax1.set_ylabel(
            r"CDF($r^2$, t$_{{lag}}$={:0.2f}s)".format(currLagTime), fontsize=15
        )
        # Set the ticks to be inside the plot area
        ax1.tick_params(which="both", direction="in")
        # Set the right-side y-axis tick marks with no numbers
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_ticks_position("both")
        # Set the axes scaling
        ax1.set_xscale("log")
        # Set location of legend on plot
        ax1.legend(loc="upper left")
        # Set the min/max values for the x, y axes
        x_min = 0.0001
        x_max = 0.1
        y_min_CDF = 0
        y_max_CDF = 1
        # Padding value for even adjustments
        axes_padding_Xaxis = 0.1
        axes_padding_CDF_Yaxis = 0.1
        # Adjust any axes below with padding, comment out what you dont want or need
        x_min = x_min - (x_min * axes_padding_Xaxis)
        x_max = x_max + (x_max * axes_padding_Xaxis)
        y_min_CDF = y_min_CDF - (y_min_CDF * axes_padding_CDF_Yaxis)
        y_max_CDF = y_max_CDF + (y_max_CDF * axes_padding_CDF_Yaxis)
        # Apply the min-max axes values to the plot
        ax1.set(ylim=(y_min_CDF, y_max_CDF), xlim=(x_min, x_max))

        # Plot the CDF1_residuals
        ax0.plot(
            CDF_x_data, CDF1_residuals, color="C0", linestyle="dashed", label="Single"
        )
        # Plot the CDF2_residuals
        ax0.plot(
            CDF_x_data, CDF2_residuals, color="C1", linestyle="dashed", label="Double"
        )
        # Set the label for the y-axis
        ax0.set_ylabel(r"Residuals", fontsize=15, labelpad=1)
        # Set the ticks to be inside the plot area
        ax0.tick_params(which="both", direction="in")
        # Set the right-side y-axis tick marks with no numbers
        ax0.yaxis.set_ticks_position("both")
        ax0.xaxis.set_ticks_position("both")
        # Plot a horizontal line at y=0 on the residuals plot for reference
        ax0.axhline(y=0.0, color="black", alpha=0.7, linestyle="dashed")
        ax0.legend(loc="upper left")
        # Adjust the y-axes for this plot here
        y_min_resid = -0.10
        y_max_resid = 0.10
        # Padding value for even adjustments
        axes_padding_resid_Yaxis = 0.1
        y_min_resid = y_min_resid + (y_min_resid * axes_padding_resid_Yaxis)
        y_max_resid = y_max_resid + (y_max_resid * axes_padding_resid_Yaxis)
        # Apply the min-max axes to the plot
        ax0.set(ylim=(y_min_resid, y_max_resid))

        # Set the title for the window
        fig.canvas.set_window_title(f"Figure {plotNum} of {outputPlotLagRange}")
        # Set a title for the plot
        fig.suptitle(
            r"CDF Plots with Residuals at Lag {:0.2f}s".format(currLagTime),
            y=0.95,
            fontsize=18,
        )

        # Adjust the vertical distance between the subplots
        plt.subplots_adjust(hspace=0.05)

        # Show the plot
        plt.show()

    # Plot CDF Weighting factors
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    # Plot wVals vs lagt, set label for legend
    ax3.plot(wValsMob_DF.index, wValsMob_DF["wVal"], "o", label=r"CDF w Values")
    # Plot a horizontal line at y=0.5 on the wVals plot for reference
    ax3.axhline(y=0.5, color="black", alpha=0.7, linestyle="dashed")
    # Set the label for the x-axis and y-axis
    ax3.set_xlabel(r"Lag Time (s)", fontsize=15)
    ax3.set_ylabel(r"CDF Weighting Factor, w", fontsize=15)
    # Set the ticks to be inside the plot area
    ax3.tick_params(which="both", direction="in")
    # Set the right-side y-axis tick marks with no numbers
    ax3.yaxis.set_ticks_position("both")
    ax3.xaxis.set_ticks_position("both")
    # Set the axes scaling
    ax3.set_xscale("linear")
    ax3.set_yscale("linear")
    # Set the min/max values for the x, y axes
    x_min_wVals = 0.0
    x_max_wVals = wValsMob_DF.index[-1]
    y_min_wVals = 0.0
    y_max_wVals = 1.0
    # Padding value for even adjustments
    axes_padding_wVals = 0.1
    # Adjust any axes below with padding, comment out what you dont want or need
    # x_min_wVals = x_min_wVals - (x_max_wVals * 0.05)
    x_max_wVals = x_max_wVals + (x_max_wVals * 0.05)
    y_min_wVals = y_min_wVals - axes_padding_wVals / 2
    y_max_wVals = y_max_wVals + axes_padding_wVals / 2
    # ! Turn on the minor tick marks on the axes
    plt.minorticks_on()
    # Apply the min-max axes values to the plot
    ax3.set(ylim=(y_min_wVals, y_max_wVals), xlim=(x_min_wVals, x_max_wVals))
    # Set the x-axis major ticks to be multiples of 2 * frameTime
    ax3.xaxis.set_major_locator(MultipleLocator(2 * frameTime / 100))
    # Set the x-axis minor ticks to be multiples of frameTime
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    # Set the title for the window
    fig2.canvas.set_window_title("CDF Weighting Factor vs Lag Time")
    # Set a title for the plot
    fig2.suptitle(
        r"Mobile CDF Weighting Factors vs Lag Time up to {}".format(
            wValsMob_DF.index[-1]
        ),
        y=0.95,
        fontsize=18,
    )
    # Show the plot
    plt.show()

    # ! Fit CDF to different models
    # !     --Random Fractal
    # !     --Geometric Fractal
    # !     --CTRW
    # !      --Any Others?!

    # TODO FIX LABELS FOR cdfONEmob-cdfTWOmob and cdfRandFract-cdfTWOmob plots so they have better titles and are more distinct
    # ! THE RAND FRACT FUNCTION SEEMS REALLY OFF COMPARED TO MATHEMATICA....
    # Random fractal modeled by percolation cluster
    # Define static variables
    # -- Anomalous diffusion exponent
    cdfRandFract_d_w = 0.423
    # -- Dimensionality of the diffusion
    cdfRandFract_dim = 2.0
    # -- u = 1.65 +- 0.1 (from a fit, see reference --Mathematica)
    cdfRandFract_u = 1.65
    # -- t -> 1.0, defined in for loop since it varies per sample
    cdfRandFract_t = 1.0
    for plotNum in range(1, outputPlotLagRange + 1):
        # Grab the columns of with the current lagtime
        plotCDF_DF = cumulDistrib_DF[[f"r2_lag{plotNum}", f"CDF_lag{plotNum}"]].copy()
        # Sort by the r2 column
        plotCDF_DF = plotCDF_DF.sort_values(
            [f"r2_lag{plotNum}", f"CDF_lag{plotNum}"], ascending=[True, False]
        )
        # Remove any trailing NaNs from the DF
        plotCDF_DF.dropna(axis=0, how="all", inplace=True)
        # Plot the two columns with r2 as 'x' and CDF as 'y'
        # Gather the x y data
        CDF_x_data = plotCDF_DF[f"r2_lag{plotNum}"]
        CDF_y_data = plotCDF_DF[f"CDF_lag{plotNum}"]
        CDF_sqrt_y_data = np.sqrt(CDF_y_data)
        # Make a custom CDF_RandFract function that uses the constants we defined, i.e. cdfRandFract_d_w, cdfRandFract_dim, cdfRandFract_u, cdfRandFract_t
        func_cdfConstRandFract = lambda x, d_f: func_cdfRandFract(
            x, d_f, cdfRandFract_dim, cdfRandFract_d_w, cdfRandFract_u, cdfRandFract_t
        )
        # Generate fitted curve and residual for func_cdfConstRandFract
        CDF_RandFract_popt, CDF_RandFract_pcov = curve_fit(
            func_cdfConstRandFract, CDF_x_data, CDF_sqrt_y_data
        )
        # Generate the residuals for the fit to the func_cdfConstRandFract function
        CDF_RandFract_residuals = CDF_y_data - func_cdfConstRandFract(
            CDF_x_data, CDF_RandFract_popt
        )
        # Generate fitted curve and residuals for cdfTWOmob
        CDF2_popt, CDF2_pcov = curve_fit(
            func_cdfTWOmob, CDF_x_data, CDF_y_data, p0=[0.5, 0.01, 0.1]
        )
        CDF2_residuals = CDF_y_data - func_cdfTWOmob(CDF_x_data, *CDF2_popt)
        # Setup the new figure for CDF plots
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(6, 9), gridspec_kw={"height_ratios": [4, 8]}, sharex=True
        )
        # Plot original data
        ax1.plot(CDF_x_data, CDF_y_data, label="Data", color="black")
        # Plot the curve for cdfRandFract
        ax1.step(
            CDF_x_data,
            func_cdfConstRandFract(CDF_x_data, *CDF_RandFract_popt),
            color="C0",
            label="Single",
            linestyle="dashed",
        )
        # Plot the curve for cdfTWOmob
        ax1.step(
            CDF_x_data,
            func_cdfTWOmob(CDF_x_data, *CDF2_popt),
            color="C1",
            label="Double",
            linestyle="dashed",
        )

        # Determine current lagTime rounded to 2 decimal places
        currLagTime = round(plotNum / frameTime, 2)
        # Set the label for the x-axis and y-axis
        ax1.set_xlabel(r"$r^2$ ($\mu$m$^2$)", fontsize=15)
        ax1.set_ylabel(
            r"CDF($r^2$, t$_{{lag}}$={:0.2f}s)".format(currLagTime), fontsize=15
        )
        # Set the ticks to be inside the plot area
        ax1.tick_params(which="both", direction="in")
        # Set the right-side y-axis tick marks with no numbers
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_ticks_position("both")
        # Set the axes scaling
        ax1.set_xscale("linear")
        # Set location of legend on plot
        ax1.legend(loc="upper left")
        # Set the min/max values for the x, y axes
        x_min = 0.0001
        x_max = 0.1
        y_min_CDF = 0
        y_max_CDF = 1
        # Padding value for even adjustments
        axes_padding_Xaxis = 0.1
        axes_padding_CDF_Yaxis = 0.1
        # Adjust any axes below with padding, comment out what you dont want or need
        x_min = x_min - (x_min * axes_padding_Xaxis)
        x_max = x_max + (x_max * axes_padding_Xaxis)
        y_min_CDF = y_min_CDF - (y_min_CDF * axes_padding_CDF_Yaxis)
        y_max_CDF = y_max_CDF + (y_max_CDF * axes_padding_CDF_Yaxis)
        # Apply the min-max axes values to the plot
        ax1.set(ylim=(y_min_CDF, y_max_CDF), xlim=(x_min, x_max))

        # Plot the CDF_RandFract_residuals
        ax0.plot(
            CDF_x_data,
            CDF_RandFract_residuals,
            color="C0",
            linestyle="dashed",
            label="Single",
        )
        # Plot the CDF2_residuals
        ax0.plot(
            CDF_x_data, CDF2_residuals, color="C1", linestyle="dashed", label="Double"
        )

        # Set the label for the y-axis
        ax0.set_ylabel(r"Residuals", fontsize=15, labelpad=1)
        # Set the ticks to be inside the plot area
        ax0.tick_params(which="both", direction="in")
        # Set the right-side y-axis tick marks with no numbers
        ax0.yaxis.set_ticks_position("both")
        ax0.xaxis.set_ticks_position("both")
        # Plot a horizontal line at y=0 on the residuals plot for reference
        ax0.axhline(y=0.0, color="black", alpha=0.7, linestyle="dashed")
        ax0.legend(loc="upper left")
        # Adjust the y-axes for this plot here
        y_min_resid = -0.10
        y_max_resid = 0.10
        # Padding value for even adjustments
        axes_padding_resid_Yaxis = 0.1
        y_min_resid = y_min_resid + (y_min_resid * axes_padding_resid_Yaxis)
        y_max_resid = y_max_resid + (y_max_resid * axes_padding_resid_Yaxis)
        # Apply the min-max axes to the plot
        ax0.set(ylim=(y_min_resid, y_max_resid))

        # Set the title for the window
        fig.canvas.set_window_title(f"Figure {plotNum} of {outputPlotLagRange}")
        # Set a title for the plot
        fig.suptitle(
            r"CDF Plots with Residuals at Lag {:0.2f}s".format(currLagTime),
            y=0.95,
            fontsize=18,
        )

        # Adjust the vertical distance between the subplots
        plt.subplots_adjust(hspace=0.05)

        # Show the plot
        plt.show()


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
    # Paths to MSD .json files to load as dataframes
    # Selected Tracks DF
    jsonSelectedTracksLoadPath = r"/home/vivek/Desktop/Piezo1 Test Data/Python_outputs/Selected_tracks/selected_track_list.json"
    # TAMSD of ALL Tracks DF (Trapped and Mobile)
    jsonTAMSDLoadPath = r"/home/vivek/Desktop/Piezo1 Test Data/Python_outputs/Statistics/MSDs/TAMSD.json"
    # Dict -List of Mobile and Trapped Tracks
    jsonMobileTrappedDictPath = r"/home/vivek/Desktop/Piezo1 Test Data/Python_outputs/Statistics/MSDs/Mobile_Trapped_tracks.json"
    # ALL Tracks ALL Lags DF (Trapped and Mobile)
    jsonAllTracksAllLags = r"/home/vivek/Desktop/Piezo1 Test Data/Python_outputs/Statistics/MSDs/All_Lagtimes.json"

    # Path to main directory for saving outputs
    savePath = r"/home/vivek/Desktop/Piezo1 Test Data/Python_outputs"

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
    # bounding indices for tracks to fit, select linear region
    fit_range = [1, 20]
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
    alphaValAnalysis(TAMSD_DF, MobileTrappedTracks_Dict["Mobile"], 20)

    # Cumulative Distributon Function, lagtime_limit is in units of number of frames and inclusive
    # Default lagtime_limit and outputPlotLagRange are 10
    cumulDistrib(
        AllTracksAllLags_DF,
        MobileTrappedTracks_Dict["Mobile"],
        frameTime,
        lagtime_limit=10,
        outputPlotLagRange=5,
    )

    # * -----  END SUBROUTINE  ----- * #

    # ! ----- START DEBUGGING  ----- ! #

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
