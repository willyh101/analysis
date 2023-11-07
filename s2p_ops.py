default_ops = {
    # general
    'diameter': 10,
    # 'fast_disk': 'k:/tmp/s2p_python',
    'do_bidiphase': False,
    'save_mat': False,
    'save_NWB': False,
    'tau': 1.0,
    # 'preclassify': 0., # apply classifier before signal extraction with a prob of 0.3
    'combined': False,
    
    # registration
    'do_registration': True, # 2 forces re-registration
    'keep_movie_raw': False, # must be true for 2 step reg
    'two_step_registration': False,
    'nimg_init': 1000, # subsampled frames for finding reference image
    'batch_size': 500, #2000, # number of frames per batch, default=500
    'align_by_chan': 1, # 1-based, use 2 for tdT
    
    # non rigid registration settings
    'nonrigid': False, # whether to use nonrigid registration
    
    # cell extraction
    'denoise': False,
    'threshold_scaling': 2.0, # adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
    'sparse_mode': False,
    'max_iterations': 50, # usualy stops at threshold scaling, default 20
    'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P), default 100
    # 'classifier_path': 'c:/users/will/code/suite2p/suite2p/classifiers/classifier_8m.npy',
    
    # deconvolution settings
    'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
    'win_baseline': 60.,  # window for maximin
    'sig_baseline': 10.,  # smoothing constant for gaussian filter
    'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
    'neucoeff': 0.7,  # neuropil coefficient
    
    # custom settings
    'remove_artifacts': (20, 512-20)
}

ops_8m_mods = {
    # general
    'diameter': 10, # shrink the diameter for ST 8m
    'tau': 0.05, # 8m is quite fast
    
    # registration
    'do_registration': 2, # 2 forces re-registration
    
    # cell extraction
     'threshold_scaling': 0.6,
}

ops_8m = {**default_ops, **ops_8m_mods}