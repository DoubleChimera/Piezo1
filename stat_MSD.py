import codecs
import json

# ! This function should take in a file path for a .json file of selected tracks and return a nested
# ! numpy array with all the tracks for further processing.  Include some documentation or comments about
# ! how to parse over the resulting array to extract tracks/values of interest.
class json_loader(file_path):
    def json_loader(self, file_path):
        objLoad = codecs.open(file_path, 'r', encoding='utf-8').read()
        lstnan = np.array(json.loads(objLoad))




if '__name__' == '__main__':
    fileLoadPath = ''
    savePath = ''
    frameTime = ''

# * #################### CURRENT DEBUGGING CODE IS BELOW ####################

# # To "unjsonify" this data use the following:
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)

# ! ####################   OLD DEBUGGING CODE IS BELOW   ####################