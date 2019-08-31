import codecs
import json

import numpy as np

# import os.path
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


def mobile_trapped_tracks(tracksDF):
    # Use some criteria to define mobile vs trapped tracks
    # Sort the two groups and output as .json
    # Plot the resulting data with best fit lines
    return None


if __name__ == "__main__":

    # * -----USER INPUTS BELOW----- * #
    # Paths to .json files to load as dataframes
    jsonSelectedTracksLoadPath = r"/home/vivek/Documents/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"
    jsonTAMSDLoadPath = (
        r"/home/vivek/Documents/Piezo1/temp_outputs/Statistics/MSDs/TAMSD.json"
    )
    jsonEAMSDLoadPath = (
        r"/home/vivek/Documents/Piezo1/temp_outputs/Statistics/MSDs/EAMSD.json"
    )
    jsonAllDisplacementsLoadPath = r"/home/vivek/Documents/Piezo1/temp_outputs/Statistics/MSDs/All_lag_displacements_microns.json"

    # Path to main directory for saving outputs
    savePath = r"/home/vivek/Documents/Piezo1/temp_outputs"

    # Experimental parameters
    pixelWidth = 0.1092  # in microns
    frameTime = 100  # in milliseconds, typical value is 50 or 100

    # Range of data to fit to a line
    fit_range = [1, 15]  # bounding indices for tracks to fit
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
    AllDisplacements_DF = pd.read_json(jsonAllDisplacementsLoadPath, orient="split")

    # * -----END   SUBROUTINE----- * #

    # ! -----DEBUGGING CODE START----- ! #

    print(selectedTracks_DF)

    # ! -----DEBUGGING CODE   END----- ! #


# --------------------------------------------------------------------------------------------------
# // TODO 1.0   Load a .json of all tracks
# TODO 2.0   Use selection criterion to differentiate mobile and trapped tracks
# TODO 3.0   Plot Mobile and Trapped Tracks with output
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
