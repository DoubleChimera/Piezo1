import file_loader as fl

if __name__ == '__main__':
    filename = r'/home/vivek/Tobias_Group/Single_Particle_Track_Piezo1/Piezo1 Trajectory for Analysis/2018_Nov_tirfm_tdtpiezo_5sec/93_2018_11_20_TIRF_mnspc_tdt_memdye_C_3_MMStack_Pos0.ome.json'
    save_path = r'/home/vivek/Python_Projects/Piezo1_MathToPython_Atom/temp'
    minfrm = 20
    txy_pts, tracks = fl.open_tracks(filename)
    lst, lstnan, trackOrigins = fl.gen_indiv_tracks(save_path, minfrm, tracks, txy_pts)