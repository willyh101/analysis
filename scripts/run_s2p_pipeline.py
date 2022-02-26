import numpy as np
from suite2p.run_s2p import run_s2p
import os
import shutil
import glob
from ScanImageTiffReader import ScanImageTiffReader

# data location and inputs
animalid = 'w32_2'
date = '20210419'
expt_ids = ['2','3','5']
result_base = 'E:/functional connectivity'
diameter = 10 # changed from 11... 1/9/18

# set your options for running
# overwrites the run_s2p.default_ops
# https://mouseland.github.io/suite2p/_build/html/settings.html
ops = {
        'fast_disk': '', # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
        'save_path0': [], # stores results, defaults to first item in data_path
        'delete_bin': True, # whether to delete binary file after processing
        # main settings
        'nplanes' : 3, # each tiff has these many planes in sequence
        'nchannels' : 2, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'diameter': 10, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1, # this is the main parameter for deconvolution
        'fs': 6.36,  # sampling rate (total across planes)
        'force_sktiff': False, # whether or not to use scikit-image for tiff reading, DEFAULT = FALSE
        # bidirectional phase offset
        'do_bidiphase': True,
        'bidiphase': 0,
        # output settings
        'save_mat': True, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        'preclassify': 0., # apply classifier before signal extraction with probability 0.3
        'aspect': 1.0, # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
        # registration settings
        'do_registration': True, # whether to register data
        'keep_movie_raw': True,
        'nimg_init': 500, # subsampled frames for finding reference image # was 300 1/20/20
        'batch_size': 2000, # number of frames per batch, default=500, formerly 5000 (WH)
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False, # whether to save registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma_time' : 0, # gaussian smoothing in time
        'smooth_sigma': 1.15, # ~1 good for 2P recordings, recommend >5 for 1P recordings
        'two_step_registration': True,
        'th_badframes': 1.0, # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,
        # non rigid registration settings
        'nonrigid': False, # whether to use nonrigid registration
        'block_size': [256, 256], # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2, # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5, # maximum pixel shift allowed for nonrigid, relative to rigid
        # cell detection settings
        'roidetect': True, # whether or not to run ROI extraction
        'spatial_scale': 0, # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5000, # max number of binned frames for the SVD
        'nsvd_for_roi': 2000, # max number of SVD components to keep for ROI detection
        'max_iterations': 50, # maximum number of iterations to do cell detection. was 50
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1, # adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement was 0.75 (WH)
        'high_pass': 100, # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        'nbinned': 5000, # (int, default: 5000) maximum number of binned frames to use for ROI detection.
        'smooth_masks': True, # whether to smooth masks in final pass of cell detection. This is useful especially if you are in a high noise regime.
        # ROI extraction parameters
        'inner_neuropil_radius': 2, # num  ber of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil, CHANGED TO 500 11/14/20 FROM 350 (DEFAULT) -WH
        'sparse_mode': False, # whether or not to run sparse_mode, default True, idk what it does
        # deconvolution settings
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        # artifact removal
        'remove_artifacts': (110,512-110)
      }


def process_data(animalid, date, expt_ids, result_base,
                 diameter=8, ext_db=None, **kwargs):
    fast_disk = 'E:/s2p_scratch'
    raw_base = 'D:/Frankenrig/Experiments/'

    if not ext_db:
        db = prepare_db(animalid, date, expt_ids, raw_base, result_base, fast_disk, diameter, **kwargs)
    else:
        print('using external db.')
        db = ext_db

    db.update(kwargs)

    try:
        shutil.rmtree(fast_disk + '/suite2p')
        print('fast disk contents deleted.')
    except:
        print('fast disk location empty.')

    print('starting suite2p...')

    opsEnd = run_s2p(ops=ops,db=db)


def prepare_db(animalid, date, expt_ids, raw_base, result_base, fast_disk, diameter):

    save_path0 = os.path.join(result_base, animalid, date, '_'.join(expt_ids))
    data_path = [os.path.join(raw_base, animalid, date, lbl) for lbl in expt_ids]

    with ScanImageTiffReader(glob.glob(data_path[0]+'/*.tif')[0]) as reader:
        metadata = reader.metadata()

    print('getting metadata from tiffs...')
    # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    db = {
          'h5py': [], # a single h5 file path[p]
          'h5py_key': 'data',
          'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs,
          'save_path0': save_path0,
          'data_path': data_path, # a list of folders with tiffs
                                                 # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
          'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
          'fast_disk': fast_disk, # string which specifies where the binary file will be stored (should be an SSD)
          'nchannels': len(metadata.split('channelSave = [')[1].split(']')[0].split(';')),
          'nplanes': len(metadata.split('hStackManager.zs = [')[1].split(']')[0].split(' ')),
          'diameter': diameter,
          'fs': float(metadata.split('scanVolumeRate = ')[1].split('\n')[0])
        }


    print(f"frame rate is {db['fs']}")
    print(f"number of channels is {db['nchannels']}")
    print(f"diameter is set to {db['diameter']}")
    return db

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    # process_data(*sys.argv[1:])
    process_data(animalid, date, expt_ids, result_base, diameter)
