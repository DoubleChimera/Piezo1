import track_selector as tracksel
import file_loader as fl

# TODO using the trackList of good track indices, make a new array of an array of all the good tracks, use lstnan
# TODO plot the tracks on top of the .tif file
# TODO (maybe) make the tracks clickable to toggle them, in case one or two were accidentally included that cross the cell membrane boundary... (maybe not... too much work... could just start over for now)
# TODO output the "Selected Tracks" as individual text files to the "Selected Tracks" folder
# TODO Use a press-enter-to-close command for the plot of tracks on the .tif, afterthis, move onto the statistical methods section of the code

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = r'/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Piezo1 Trajectory for Analysis/2018_Nov_tirfm_tdtpiezo_5sec/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    save_path = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    tifFile = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/test_images/test1.tiff'
    minfrm = 20

    txy_pts, tracks = fl.open_tracks(filename)
    lst, lstnan, trackOrigins = fl.gen_indiv_tracks(save_path, minfrm, tracks, txy_pts)

    xvals, yvals = tracksel.SelectFromCollection.select_tracks_plot(trackOrigins)

    img = plt.imread(tifFile)

    subplot_kw = dict(xlim=(0, 1024), ylim=(1024, 0), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(xvals, yvals, s=5, c='chartreuse')
    selector = tracksel.SelectFromCollection(ax, pts)
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

    trackList = tracksel.memBoundTracks(trackOrigins, lassoPoints)

    print(trackList)
