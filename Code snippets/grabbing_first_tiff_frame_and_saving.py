# Opens a tiff file, grabs the first frame, makes a copy, and saves it as a tiff file

import PIL
from PIL import Image
im = Image.open('/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Training March 21 2019/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_5_MMStack_Pos0.ome.tif').copy()
im = im.convert("RGB").save('/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/test_images/test1.tiff')
