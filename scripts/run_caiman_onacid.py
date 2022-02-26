import sys
from pathlib import Path

sys.path.append('H:/My Drive/Code/holofun')
# sys.path.append('c:/users/will/code/analysis')

from holofun.tiffs import SItiff
from holofun.caiman_utils import run_caiman_onacid
from holofun.utils import tic, ptoc_min

# input params
# data_root = 'd:/frankenrig/experiments'
data_root = 'x:/will/scanimage data'
mouse = 'w42_2'
date = '20220208'
epochs = ['5']
xtrim = 106

# output params
# results_root = 'd:/frankenrig/experiments'
results_root = 'e:/results'

# other params
caiman_temp = 'k:/tmp/caiman'

# pths = [Path(data_root, mouse, date, epoch) for epoch in epochs]
pths = [Path(data_root, date, mouse, epoch) for epoch in epochs]

movs = list(pths[0].glob('*.tif*'))
tmp_tiff = SItiff(movs[0])
nplanes = tmp_tiff.nplanes
fr  = tmp_tiff.fr

opts = {
    'src_folders': pths,
    'results_root': results_root,
    'mouse': mouse,
    'date': date,
    'epochs': epochs,
    'caiman_temp': Path(caiman_temp),
    'xtrim': xtrim,
    'nplanes': nplanes
}

# caiman params
dxy = (1.5, 1.5)      # spatial resolution in x and y in (um per pixel)
max_shift_um = (12., 12.)       # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])

params = {
    # MOTION CORRECTION
   'fr': fr,  # imaging rate in frames per second
   'dxy': dxy,  # spatial resolution in x and y in (um per pixel)
   'pw_rigid': False, # flag to select rigid vs pw_rigid motion correction
   'niter_rig': 2,
   'max_shifts': max_shifts, # maximum shift in um
   'strides': strides, # create a new patch every x pixels for pw-rigid correction
   'overlaps': (24, 24), # overlap between pathes (size of patch strides+overlaps)
   'max_deviation_rigid': 3, # maximum deviation allowed for patch with respect to rigid shifts
   'border_nan': 'copy', # replicate values along the boundary (if True, fill in with NaN)
   'shifts_opencv': True,  # flag for applying shifts using cubic interpolation (otherwise FFT)
   'gSig_filt': None, # (3,3) size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed
   'use_cuda': False,
   
    # SOURCE EXTRACTION
    'p': 1, # order of the autoregressive system
    'nb': 2, # number of global background components
    'rf': 25, # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    'stride': 6, # amount of overlap between the patches in pixels (CNMF)
    'K': 4, # number of components per patch
    'rolling_sum': True,
    'only_init': True,
    'ssub': 1, # spatial subsampling during initialization
    'tsub': 1, # temporal subsampling during intialization
    'merge_thr': 0.85, # merging threshold, max correlation allowed
    'min_SNR': 2.0, # signal to noise ratio for accepting a component
    'rval_thr': 0.85, # space correlation for accepting a component
    'use_cnn': True,
    'min_cnn_thr': 0.9, # threshold for CNN based classifier
    'cnn_lowest': 0.1, # neurons with cnn probability lower than this are rejected
    'gSig': (5,5),
    'decay_time': 1,  # sensor tau
    
    # ONACID
    'init_method': 'bare', # or 'cnmf'
    'init_batch':  200,
    'motion_correct': True,
    'expected_comps': 500,
    'update_num_comps': True,
    'dist_shape_update': True,
    'normalize': True,
    'sniper_mode': True,
    'test_both': False,
    'ring_CNN': False,
    'simultaneously': True,
}


if __name__  == '__main__':
    tt = tic()
    for p in range(nplanes):
        run_caiman_onacid(opts, params, p)
    ptoc_min(tt)
    print('All done!')