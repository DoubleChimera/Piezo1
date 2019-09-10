import codecs
import json
import os.path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

import file_loader as fl


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.1):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def select_tracks_plot(trackOrigins):
        xyOrigins = []
        for xycoords in trackOrigins.values():
            xyOrigins.append(xycoords)
        xvals = [coord[0] for coord in xyOrigins]
        yvals = [coord[1] for coord in xyOrigins]
        return xvals, yvals

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def memBoundTracks(trackOrigins, lassoPoints):
    trackList = []
    trackList = [
        index for index, coords in trackOrigins.items() if coords in lassoPoints
    ]
    timestr = time.strftime("%Y%m%d-%H:%M:%S")
    SelecTracksDir = os.path.join(save_path, "Selected_tracks")
    if not os.path.exists(SelecTracksDir):
        os.makedirs(SelecTracksDir)
    df = pd.DataFrame(trackList)
    completeName = os.path.join(SelecTracksDir, "selected_tracks_indices.txt")
    df.to_csv(completeName, index=False, header=[f"Selected Track Index -- {timestr}"])
    return trackList


def genSelectedTrackList(allTracks, selectedTrackIndices, img, save_path):
    selectedTrackList = []
    for index in selectedTrackIndices:
        selectedTrackList.append(allTracks[index])
    plt.figure(figsize=(10, 10))
    SelecTracksDir = os.path.join(save_path, "Selected_tracks")
    if not os.path.exists(SelecTracksDir):
        os.makedirs(SelecTracksDir)
    # dump the whole tracklist as a json
    for index, track in enumerate(selectedTrackList):
        plt.plot(track[:, 1], track[:, 2], color="chartreuse")
        df = pd.DataFrame(track)
        completeName = os.path.join(
            SelecTracksDir, f"Track{selectedTrackIndices[index]}.txt"
        )
        df.to_csv(
            completeName,
            index=False,
            header=["Frame_Number", "X-coordinate", "Y-coordinate"],
        )
    plt.imshow(img)
    plt.suptitle(
        "Close plot to continue...",
        x=0.40,
        y=0.95,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=15,
    )
    plt.show()
    # Outputs a .json file with all the selected tracks to a specified directory
    outSelTracksDir = os.path.join(SelecTracksDir, "selected_track_list.json")
    json.dump(
        selectedTrackList,
        cls=NumpyEncoder,
        fp=codecs.open(outSelTracksDir, "w", encoding="utf-8"),
        separators=(",", ":"),
        sort_keys=True,
    )
    return selectedTrackList


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class imgPreProcess(object):
    def isDicFile(self, imgPath):
        if "_dic_" in imgPath:
            img = cv2.imread(imgPath)
            im2 = img.copy()
            im2[:, :, 0] = img[:, :, 2]
            im2[:, :, 2] = img[:, :, 0]
            return im2
        else:
            img = plt.imread(imgPath)
            return img


class trackPlots(object):
    def lassoPlot(self, filename, save_path, minfrm):
        self.txy_pts, self.tracks = fl.open_tracks(filename)
        self.lst, self.lstnan, self.trackOrigins = fl.gen_indiv_tracks(
            save_path, minfrm, self.tracks, self.txy_pts
        )
        self.xvals, self.yvals = SelectFromCollection.select_tracks_plot(
            self.trackOrigins
        )

        self.img = iPP.isDicFile(tifFile)

        self.subplot_kw = dict(xlim=(0, 1024), ylim=(1024, 0), autoscale_on=False)
        self.fig, self.ax = plt.subplots(subplot_kw=self.subplot_kw, figsize=(10, 10))

        self.pts = self.ax.scatter(self.xvals, self.yvals, s=5, c="chartreuse")
        self.selector = SelectFromCollection(self.ax, self.pts)
        self.imgplot = plt.imshow(self.img)

    def plotReturn(self):
        self.fig.canvas.mpl_connect("key_press_event", tP.accept)
        self.ax.set_title("Press enter to accept selected points.")
        plt.show()
        self.lassoPoints = self.selector.xys[self.selector.ind]
        return self.lassoPoints, self.trackOrigins, self.lstnan, self.img

    def accept(self, event):
        if event.key == "enter":
            self.selector.disconnect()
            self.ax.set_title("")
            plt.close()


if __name__ == "__main__":
    # * USER INPUTS GO BELOW * #
    filename = r"/home/vivek/Documents/Python Programs/Piezo1/Piezo1/Practice data/AL_12_2019-05-30-TIRFM_Diff_tdt-mNSPCs_1_max5px_gap0.json"
    tifFile = r"/home/vivek/Documents/Python Programs/Piezo1/Piezo1/Practice data/AL_12_2019-05-30-TIRFM_Diff_tdt-mNSPCs_1_dic_MMStack_Pos0.ome.tif"
    save_path = r"/home/vivek/Documents/Python Programs/Piezo1/temp_outputs"
    minfrm = 50
    # * END OF USER INPUTS * #

    tP = trackPlots()
    iPP = imgPreProcess()

    tP.lassoPlot(filename, save_path, minfrm)

    lassoPoints, trackOrigins, lstnan, img = tP.plotReturn()

    trackList = memBoundTracks(trackOrigins, lassoPoints)

    selectedTrackList = genSelectedTrackList(lstnan, trackList, img, save_path)
