# Editing Lasso Example to be track selector, will remain to track_selector.py when done

import numpy as np
import file_loader as fl

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

# TODO define a function that returns the good list of tracks
def memBoundTracks(trackOrigins, lassoPoints):
    trackList = []

    return trackList

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = r'/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Piezo1 Trajectory for Analysis/2018_Nov_tirfm_tdtpiezo_5sec/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    save_path = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    minfrm = 20
    txy_pts, tracks = fl.open_tracks(filename)
    lst, lstnan, trackOrigins = fl.gen_indiv_tracks(save_path, minfrm, tracks, txy_pts)

    tifFile = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/test_images/test1.tiff'
    xvals, yvals = SelectFromCollection.select_tracks_plot(trackOrigins)

    img = plt.imread(tifFile)

    subplot_kw = dict(xlim=(0, 1024), ylim=(1024, 0), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

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
    print("Selected track indices:")
    print(trackList)