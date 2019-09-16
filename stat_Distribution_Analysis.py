import pandas as pd


def loadMobileTracksDF(selectedTracks_DF, mobileTrack_List, savePath):
    mobileTrack_DF = pd.DataFrame()
    for particle, track in selectedTracks_DF.reset_index(drop=True).groupby("particle"):
        if particle in mobileTrack_List:
            mobileTrack_DF = mobileTrack_DF.append(track, ignore_index=True)
    return mobileTrack_DF


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
    # Paths to MSD .json files to load as dataframes
    # Selected Tracks DF
    jsonSelectedTracksLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"
    # TAMSD of ALL Tracks DF (Trapped and Mobile)
    jsonTAMSDLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/TAMSD.json"
    # EAMSD of ALL Tracks DF (Trapped and Mobile)
    jsonEAMSDLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs/EAMSD.json"
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

    # * ----- START SUBROUTINE ----- * #

    # * -----  END SUBROUTINE  ----- * #

    # ! ----- START DEBUGGING  ----- ! #

    # ! -----  END DEBUGGING   ----- ! #
