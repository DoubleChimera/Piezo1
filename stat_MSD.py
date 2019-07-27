import codecs
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class json_loader(object):
    def json_to_array(self, file_path):
        self.objLoad = codecs.open(file_path, 'r', encoding='utf-8').read()
        self.lstnan = np.array(json.loads(self.objLoad))
        self.arrNan = np.array([np.array(track) for track in self.lstnan])
        return self.arrNan


# * This class remains unused, and is for reference for now
class taMSD(object):
    def iMSD(self, traj, dt, with_nan=True):
        self.shifts = np.arange(1, len(traj), dtype='int')
        self.msd = np.empty((len(self.shifts), 2), dtype='float')
        self.msd[:] = np.nan

        self.msd[:, 1] = self.shifts * dt

        for i, shift in enumerate(self.shifts):
            self.diffs = traj[:-shift] - traj[shift:]
            if with_nan:
                self.diffs = self.diffs[~np.isnan(self.diffs).any(axis=1)]
            self.diffs = np.square(self.diffs).sum(axis=1)

            if len(self.diffs) > 0:
                self.msd[i, 0] = np.mean(self.diffs)

        self.msd = pd.DataFrame(self.msd)
        self.msd.columns = ["msd", "delay"]

        self.msd.set_index('delay', drop=True, inplace=True)
        self.msd.dropna(inplace=True)

        return self.msd

    def bulkMSD(self, trackArray):
        #loop over the list of tracks and run imsd, summing the results here
        # takes a track array
        # passes each array to iMSD
        # determines which result is the longest and adds subsequent results to it
        # returns that value
        return None



class MSD(object):
    def ensemble_averaged(self, tracks, ftime, pwidth, with_nan=True):
        self.tracks = tracks
        self.ftime = ftime
        maxVal = 0
        maxVal = [val.shape[0] for val in tracks if val.shape[0] > maxVal]  # This returns a list of maxVals for each track, use max(maxVal) for absolute max value
        self.ensMSD = np.zeros((max(maxVal), 1), dtype='float')

        for track in tracks:
            diff = track[:,1:3] - track[0,1:3]
            # adjust pixel values to um distances using pwidth
            diff = diff * pwidth
            diff = np.square(diff).sum(axis=1).reshape(len(diff),1)
            self.indivSum = np.nansum(np.stack((self.ensMSD[:len(diff)], diff)), axis=0)
            self.ensMSD[:len(diff)] = self.indivSum

        self.ensMSD = self.ensMSD / len(tracks)

        self.lagTimes = (np.arange(0, len(self.ensMSD), dtype='float').reshape(max(maxVal),1)) * self.ftime
        self.ensMSD = np.append(self.lagTimes, self.ensMSD, axis = 1)
        self.ensMSD = pd.DataFrame(self.ensMSD)
        self.ensMSD.columns = ["Lag Time","EAMSD"]
        # self.ensMSD.set_index('Lag Time', drop=True, inplace=True)  # * Unused property - sets an arbitrary column as the index, in this case 'Lag Time'
        self.ensMSD.dropna(inplace=True)
        # returns pandas data frame with lag time in seconds <r^2(del)> in um^2
        return self.ensMSD

    def plot_eaMSD(self, save_path, eaMSD_coords):
        # ! This is where the next code needs to go -- NEED TO ADD STDEV and ERROR CLOUD to plot!!!
        self.df = eaMSD_coords
        ax = plt.gca()

        self.df.plot(kind='line',x='Lag Time',y='EAMSD',ax=ax)
        ax.set(xlabel="Lag Time (s)", ylabel=r"MSD ($\mu$$m^2$)")
        plt.show()
        return None

if __name__ == '__main__':
    #################### * USER INPUTS BELOW * ####################
    fileLoadPath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp/Selected_tracks/selected_track_list.json'
    savePath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    # time (in ms) between frames from experiment, typically 50ms or 100ms
    pixelWidth = .1092      # in microns
    frameTime = 50          # in milliseconds

    #################### * END OF USER INPUTS * ###################

    # Instantiates and loads .json file as trackArray
    jl = json_loader()
    trackArray = jl.json_to_array(fileLoadPath)

    # Instantiates and loads MSD() class
    msd = MSD()
    # Converts time to seconds
    frameTime = frameTime / 1000

    # Returns a pandas data frame of values for EAMSD in units seconds and um^2
    ensMSD = msd.ensemble_averaged(tracks=trackArray, ftime=frameTime, pwidth=pixelWidth)
    # Plots the returned values and outputs the plot
    msd.plot_eaMSD(savePath, ensMSD)

    # Returns a pandas data frame of values for TAMSD
    # Plots the returned values and outputs the plot

    # * #################### CURRENT DEBUGGING CODE IS BELOW ####################
    print(ensMSD)

    # ! ####################   OLD DEBUGGING CODE IS BELOW   ####################
    # # Set 'r' as trackArray
    # r = trackArray # * DONE!
    # # make a copy to work with of trackArray, store as 's'   # * DO WE NEED THIS COPY????  NO!!!  DONE!
    # s = r.copy() # * DONE! IGNORED!
    # # find the length of the longest track in trackArray
    # maxVal = 0  # * DONE!
    # # this returns a list of the lengths of all the tracks
    # maxVal = [val.shape[0] for val in s if val.shape[0] > maxVal] # * DONE!
    # # make a list of zeros as long as the longest track with 2 columns for x-y
    # a = np.zeros((max(maxVal), 1), dtype='float') # * DONE!
    # # print(a)

    # # for each track
    # for track in s: # * DONE!
    # # subtrack the origin val from each frame
    #     diff = track[:,1:3] - track[0,1:3] # * DONE!
    # # square those values and sum each row
    #     diff = np.square(diff).sum(axis=1).reshape(len(diff),1) # * DONE!
    # # add thos values to the total msd matrix, column 1
    #     b = np.sum(np.stack((a[:len(diff)], diff)), axis=0) # * DONE!
    #     a[:len(diff)] = b # * DONE!
    # # divide by the number of tracks
    # a = a / len(s) # * DONE!

    # # Now add in the lag times...
    # # first assume a time per frame of 50 ms
    # frame_time = .050 # * DONE!
    # timing = (np.arange(0, len(a), dtype='float').reshape(100,1)) * frame_time # * DONE!
    # a = np.append(timing, a, axis = 1) # * DONE!
    # a = pd.DataFrame(a) # * DONE!
    # a.columns = ["Lag Time","EAMSD"] # * DONE!
    # # a.set_index('Lag Time', drop=True, inplace=True)  # * Unused - sets an arbitrary column as the index, in this case 'Lag Time'
    # a.dropna(inplace=True) # * DONE!

    # print(s[0][0,1:3])
    # print(s[0][0])
    # diff = s[0][:,1:3] - s[0][0,1:3]
    # print(diff)
    # diff = np.square(diff).sum(axis=1)
    # print(diff)
    # print(type(diff))


    # for val in s:
    #     if val.shape[0] > max:
    #         max = val.shape[0]
    #     print(max)

    # Tracks can be accessed by index, i.e. trackArray[0], trackArray[1], etc.
    # Frame index: trackArray[0][:,0]    X-coords: trackArray[0][:,1]   Y-coords: trackArray[0][:,2]  XY-coords: trackArray[0][:,1:3]
    # XY-coord of 0th frame trackArray[0][0,1:3]