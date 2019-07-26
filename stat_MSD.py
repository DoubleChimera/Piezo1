import codecs
import json
import numpy as np
import pandas as pd

# ! This function should take in a file path for a .json file of selected tracks and return a nested
# ! numpy array with all the tracks for further processing.  Include some documentation or comments about
# ! how to parse over the resulting array to extract tracks/values of interest.
class json_loader(object):
    def json_to_array(self, file_path):
        self.objLoad = codecs.open(file_path, 'r', encoding='utf-8').read()
        self.lstnan = np.array(json.loads(self.objLoad))
        self.arrNan = np.array([np.array(track) for track in self.lstnan])
        return self.arrNan


class MSD(object):
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

    def eaMSD(self, trackArray):
        #loop over the list of tracks and run imsd, summing the results here
        # takes a track array
        # passes each array to iMSD
        # determines which result is the longest and adds subsequent results to it
        # returns that value
        return None

if __name__ == '__main__':
    #################### * USER INPUTS BELOW * ####################
    fileLoadPath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp/Selected_tracks/selected_track_list.json'
    savePath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    # time (in us) between frames from experiment, typically 50us or 100us
    frameTime = '50'
    #################### * END OF USER INPUTS * ###################

    # Instantiates and loads .json file as trackArray
    # Tracks can be accessed by index, i.e. trackArray[0], trackArray[1], etc.
    # Frame index: trackArray[0][:,0]    X-coords: trackArray[0][:,1]   Y-coords: trackArray[0][:,2]  XY-coords: trackArray[0][:,1:3]
    # XY-coord of 0th frame trackArray[0][0,1:3]
    jl = json_loader()
    trackArray = jl.json_to_array(fileLoadPath)

    MSD = MSD()
    r = trackArray[1][:,1:3]

    msd = MSD.iMSD(traj=r, dt=.05, with_nan=True)
    print(msd)
    # iMSD does the individual MSD, now just need to loop over the whole set of trajectories.... should be easy... yeah right


# * #################### CURRENT DEBUGGING CODE IS BELOW ####################

# ! ####################   OLD DEBUGGING CODE IS BELOW   ####################