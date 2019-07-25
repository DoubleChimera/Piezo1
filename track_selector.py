# Editing Lasso Example to be track selector, will remain to track_selector.py when done

import numpy as np
import os.path
import pandas as pd
import file_loader as fl
import time
import json
import codecs

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


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
            raise ValueError('Collection must have a facecolor')
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
    trackList = [index for index, coords in trackOrigins.items() if coords in lassoPoints]
    timestr = time.strftime("%Y%m%d-%H:%M:%S")
    SelecTracksDir = os.path.join(save_path, 'Selected_tracks')
    if not os.path.exists(SelecTracksDir):
        os.makedirs(SelecTracksDir)
    df = pd.DataFrame(trackList)
    completeName = os.path.join(SelecTracksDir, 'selected_tracks_indices.txt')
    df.to_csv(completeName, index=False, header=[f'Selected Track Index -- {timestr}'])
    return trackList


def genSelectedTrackList(allTracks, selectedTrackIndices, tifFile, save_path):
    selectedTrackList = []
    for index in selectedTrackIndices:
        selectedTrackList.append(allTracks[index])
    img = plt.imread(tifFile)
    plt.figure(figsize=(10,10))
    SelecTracksDir = os.path.join(save_path, 'Selected_tracks')
    if not os.path.exists(SelecTracksDir):
        os.makedirs(SelecTracksDir)
    #dump the whole tracklist as a json
    for index, track in enumerate(selectedTrackList):
        plt.plot(track[:,1], track[:,2], color='chartreuse')
        df = pd.DataFrame(track)
        completeName = os.path.join(SelecTracksDir, f'Track{selectedTrackIndices[index]}.txt')
        df.to_csv(completeName, index=False, header=['Frame_Number','X-coordinate','Y-coordinate'])
    implot = plt.imshow(img)
    plt.suptitle("Close plot to continue...", x=0.40, y=.95, horizontalalignment='left', verticalalignment='top', fontsize = 15)
    plt.show()
    # Prepare / encodes selectedTrackList for .json dump
    outSelTrackList = json.dumps(selectedTrackList, cls=NumpyEncoder)
    # This outputs a .json file with all the selected tracks
    outSelTracksDir = os.path.join(SelecTracksDir, 'selected_track_list.json')
    json.dump(outSelTrackList, codecs.open(outSelTracksDir, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    return selectedTrackList

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # * USER INPUTS GO BELOW * #
    filename = r'/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Piezo1 Trajectory for Analysis/2018_Nov_tirfm_tdtpiezo_5sec/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    tifFile = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/test_images/test1.tiff'
    save_path = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    minfrm = 20
    # * END OF USER INPUTS * #

    txy_pts, tracks = fl.open_tracks(filename)
    lst, lstnan, trackOrigins = fl.gen_indiv_tracks(save_path, minfrm, tracks, txy_pts)
    xvals, yvals = SelectFromCollection.select_tracks_plot(trackOrigins)

    img = plt.imread(tifFile)

    subplot_kw = dict(xlim=(0, 1024), ylim=(1024, 0), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(10,10))

    pts = ax.scatter(xvals, yvals, s=5, c='chartreuse')
    selector = SelectFromCollection(ax, pts)
    implot = plt.imshow(img)


    def accept(event):
        if event.key == "enter":
            selector.disconnect()
            ax.set_title("")
            plt.close()


    def plotReturn():
        fig.canvas.mpl_connect("key_press_event", accept)
        ax.set_title("Press enter to accept selected points.")

        plt.show()
        lassoPoints = selector.xys[selector.ind]
        return lassoPoints


    lassoPoints = plotReturn()

    trackList = memBoundTracks(trackOrigins, lassoPoints)

    selectedTrackList = genSelectedTrackList(lstnan, trackList, tifFile, save_path)