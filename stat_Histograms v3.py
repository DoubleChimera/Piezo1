import json
import os.path
import codecs
import numpy as np
import pandas as pd


def MSD_df_to_json(savePath, df_results):
    df_results = df_results
    savePath = savePath
    # Make directory if it doesn't exist already
    outResultsDF_json = os.path.join(savePath, "Statistics/MSDs")
    if not os.path.exists(outResultsDF_json):
        os.makedirs(outResultsDF_json)
    # Determine which MSD, name .json accordingly
    if df_results.columns[0] == 0:
        outJsonName = os.path.join(outResultsDF_json, "TAMSD.json")
    elif df_results.columns[0] == "<x>":
        outJsonName = os.path.join(outResultsDF_json, "EAMSD.json")
    elif df_results.columns[0] == "particle":
        outJsonName = os.path.join(outResultsDF_json, "All_lag_displacements.json")
    # Output dataframe to .json in determined directory
    df_results.to_json(outJsonName)


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


class outputDisplacementJson(object):
    def setPosColumns(self, pos_columns=None):
        self.pos_columns = pos_columns
        if self.pos_columns is None:
            self.pos_columns = ["x", "y"]
        return self.pos_columns

    def genLagColumns(self, lag_columns, pos_columns):
        self.lag_columns = lag_columns
        self.pos_columns = pos_columns
        for lag in lag_columns:
            for p in pos_columns:
                yield "".join(map(str, (p, lag)))

    def genAllDisplacements(self, tracks, frameTime, pos_columns, savePath):
        self.tracks = tracks
        self.frameTime = frameTime
        self.pos_columns = pos_columns
        self.savePath = savePath
        # Determines max track length and gens maxLagtimes List
        self.maxLagTime = list(range(self.tracks.groupby("particle").count().max()[0]))
        # Determines total number of particles in DF
        self.totalParticleCount = self.tracks.groupby("frame").count().max()[0]

        self.tracks.insert(2, "lagt", (self.tracks["frame"] * self.frameTime))
        self.lag_columns = ["_lag{}".format(l) for l in self.maxLagTime]
        self.lag_results = list(
            outDisp.genLagColumns(self.lag_columns, self.pos_columns)
        )
        self.results = self.tracks.reindex(
            columns=self.tracks.columns.tolist() + self.lag_results
        )
        self.results.set_index("particle", inplace=True)

        for particle, track in self.tracks.reset_index(drop=True).groupby("particle"):
            self.trackLength = int(len(track["frame"]))
            for lag in range(self.trackLength):
                if lag == 0:
                    self.results.loc[particle, [f"x_lag{lag}", f"y_lag{lag}"]] = track[
                        self.pos_columns
                    ].values
                else:
                    indivDisp = pd.DataFrame(
                        track[self.pos_columns].values[lag:]
                        - track[self.pos_columns].values[:-lag],
                        columns=[[f"x_lag{lag}", f"y_lag{lag}"]],
                    )
                    # the '_' is a dummy var that is never called
                    # it is just for iteration over the loop
                    for _ in range(lag):
                        indivDisp.loc[indivDisp.index.max() + 1] = None
                    self.results.loc[
                        particle, [f"x_lag{lag}", f"y_lag{lag}"]
                    ] = indivDisp[[f"x_lag{lag}", f"y_lag{lag}"]].values

        # output the json!
        self.results.reset_index(inplace=True)
        MSD_df_to_json(self.savePath, self.results)
        # Return the results DataFrame
        return self.results


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

    # Instantiate outputDisplacementJson() class
    outDisp = outputDisplacementJson()
    # Instantiate methods/functions
    # Define pos_columns via function
    pos_columns = outDisp.setPosColumns()
    # Generate the DataFrame of displacements and output the file to the savePath
    results = outDisp.genAllDisplacements(tracks, frameTime, pos_columns, savePath)

    # ! --------------------   CURRENT DEBUGGING CODE IS BELOW   -------------------- ! #

    # ! -------   OLD DEBUGGING CODE IS BELOW   ------- ! #
