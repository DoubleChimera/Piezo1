import codecs
import json
import numpy as np

# ! This function should take in a file path for a .json file of selected tracks and return a nested
# ! numpy array with all the tracks for further processing.  Include some documentation or comments about
# ! how to parse over the resulting array to extract tracks/values of interest.
class json_loader(object):
    def json_to_array(self, file_path):
        self.objLoad = codecs.open(file_path, 'r', encoding='utf-8').read()
        self.lstnan = np.array(json.loads(self.objLoad))
        return self.lstnan



if __name__ == '__main__':
    # * USER INPUTS BELOW * #
    fileLoadPath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp/Selected_tracks/selected_track_list.json'
    savePath = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    # time (in us) between frames from experiment, typically 50us or 100us
    frameTime = '50'
    # * END OF USER INPUTS * #

    jl = json_loader()
    trackList = jl.json_to_array(fileLoadPath)
    print(trackList[0])


# * #################### CURRENT DEBUGGING CODE IS BELOW ####################

# # To "unjsonify" this data use the following:
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)

# ! ####################   OLD DEBUGGING CODE IS BELOW   ####################