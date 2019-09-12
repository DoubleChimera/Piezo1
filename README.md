# Piezo1 Single Particle Tracking
Python-based statistics and plots of single particle tracking experiments

Takes in .json data from flika and a corresponding .tif file

Instructions:   DON'T EDIT the file_loader.py file, it is a dependency of track_selector.py
                Edit the track_selector.py at "__name__" = "__main__" near end of the file, look for User Inputs
                This will output to your save path.
                In the 'Selected_tracks' folder, a .json of useable tracks will be generated
                Use this .json for all the 'stat_xxxx.py' files

                Step 1: Run track_selector.py
                Step 2: Run stat_MSD.py
                Step 3: Run mobile_and_trapped.py
                Step 4: TBD...


Current problems:   EAMSD StdDev and thus any error clouds need to be re-evaluated for accuracy... C
                    Currently using a placeholder method to move along with other aspects
