import json
import codecs
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


if __name__ == "__main__":

    # * --------------------   USER INPUTS BELOW   -------------------- * #
    savePath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs"
    jsonTracksLoadPath = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Selected_tracks/selected_track_list.json"
    loadPath = (
        r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs/Statistics/MSDs"
    )
    frameTime = 50  # in milliseconds
    pixelWidth = 0.1092  # in microns
    # * --------------------   END OF USER INPUTS   -------------------- * #

    frameTime = frameTime / 1000
    # Instantiate json_converter() class
    jc = json_converter()
    # Pull tracks into dataframe from .json file
    # These tracks use pixels and frames, not microns and seconds
    tracks = jc.json_tracks_to_df(jsonTracksLoadPath)

    # ! -------------   CURRENT DEBUGGING CODE IS BELOW   ------------- ! #

    # class genAllDisplacements(pos_columns=None):

    print(tracks)

    # ! -------   OLD DEBUGGING CODE IS BELOW   ------- ! #
