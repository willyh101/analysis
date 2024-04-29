default_ops = {
    # general
    'diameter': 10,
    # 'fast_disk': 'k:/tmp/s2p_python',
    'do_bidiphase': True,
    'save_mat': False,
    'save_NWB': False,
    'tau': 1.25,
    # 'preclassify': 0., # apply classifier before signal extraction with a prob of 0.3
    'combined': False,
    
    # registration
    'do_registration': 2, # 2 forces re-registration
    'keep_movie_raw': False, # must be true for 2 step reg
    'two_step_registration': False,
    'nimg_init': 1000, # subsampled frames for finding reference image
    'batch_size': 2000, #2000, # number of frames per batch, default=500
    'align_by_chan': 1, # 1-based, use 2 for tdT
    'smooth_sigma_time': 1., # set to 1 or 2 for low SNR data
    
    # non rigid registration settings
    'nonrigid': True, # whether to use nonrigid registration
    
    # cell extraction
    'denoise': True,
    'threshold_scaling': 0.7, # adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
    'sparse_mode': True,
    'max_iterations': 300, # usualy stops at threshold scaling, default 20
    'high_pass': 50,  # running mean subtraction with window of size 'high_pass' (use low values for 1P), default 100
    'classifier_path': '/home/will/code/suite2p/suite2p/classifiers/classifier.npy',
    'max_overlap': 0.5,  # cells with more overlap than this get removed during triage, before refinement
    'nbinned': 5000,  # number of binned frames for cell detection, default: 5000
    'spatial_scale': 2,  # spatial scale of the data, depends on microscope, 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels)
    'spatial_hp_detect': 25, # (int, default: 25) window for spatial high-pass filtering for neuropil 

    # deconvolution settings
    'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
    'win_baseline': 60.,  # window for maximin
    'sig_baseline': 10.,  # smoothing constant for gaussian filter
    'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
    'neucoeff': 0.7,  # neuropil coefficient
    
    # custom settings
    'remove_artifacts': (25, 512-25)
}

_ops_8m_mods = {
    # general
    # 'diameter': 8, # shrink the diameter for ST 8m
    'diameter': 8, # for zoom 2
    'tau': 0.1, # 8m is quite fast
    
    # registration
    'do_registration': True, # 2 forces re-registration
    'two_step_registration': False,
    
    # cell extraction
     'threshold_scaling': 0.6,

     # custom
     'remove_artifacts': (80, 512-80)
}

ops_8m = {**default_ops, **_ops_8m_mods}