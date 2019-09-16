import codecs
import json

# import os.path
import numpy as np
import pandas as pd


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
    # Convert frameTime to frames-per-second
    frameTime = 1000 / frameTime

    # Instantiate the json_converter class
    jc = json_converter()

    # Load tracks data into a pandas DataFrame
    selectedTracks_DF = jc.json_SelectedTracks_to_DF(jsonSelectedTracksLoadPath)
    TAMSD_DF = pd.read_json(jsonTAMSDLoadPath, orient="split")
    AllTracksLags_DF = pd.read_json(jsonAllTracksAllLags, orient="split")
    MobileTrappedTracks_Dict = jc.load_MobileTrapped_json(jsonMobileTrappedDictPath)

    # Setup the index for TAMSD
    TAMSD_DF.set_index("lagt", inplace=True)

    # * -----  END SUBROUTINE  ----- * #

    # ! ----- START DEBUGGING  ----- ! #

    # ! -----  END DEBUGGING   ----- ! #
