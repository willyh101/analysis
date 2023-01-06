from pathlib import Path
from glob import glob
import sys
import shutil
from suite2p import run_s2p

sys.path.append('/home/will/code/holofun')

from holofun.simple_guis import openfoldersgui
from holofun.si_tiff import SItiffCore

# assuming you are coming from the data drive... (and not the server)
tiff_base = '/mnt/data2/experiments'
result_base = '/mnt/localdata/figure ground'

default_ops = {
    # general
    'diameter': 8,
    # 'fast_disk': 'k:/tmp/s2p_python',
    'do_bidiphase': True,
    'save_mat': False,
    'save_NWB': False,
    'tau': 1.0,
    # 'preclassify': 0., # apply classifier before signal extraction with a prob of 0.3
    'combined': False,
    
    # registration
    'do_registration': True, # force re-registration
    'keep_movie_raw': True, # must be true for 2 step reg
    'two_step_registration': True,
    'nimg_init': 800, # subsampled frames for finding reference image
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
    'remove_artifacts': (100,512-100)
}

# prompt epoch folders and create paths
epoch_folders = openfoldersgui(tiff_base)
epoch_names = [Path(epoch).stem for epoch in epoch_folders]
date = Path(epoch_folders[0]).parent.stem
mouse = Path(epoch_folders[0]).parent.parent.stem
data_path = [str(Path(tiff_base, mouse, date, lbl).as_posix()) for lbl in epoch_names]
save_path = str(Path(result_base, mouse, date, '_'.join(epoch_names)).as_posix())

tmp_tiff = SItiffCore(glob(data_path[0]+'/*.tif')[0])

db = {
    'save_path0': save_path,
    'data_path': data_path,
    'nchannels': tmp_tiff.nchannels,
    'nplanes': tmp_tiff.nplanes,
    'fs': tmp_tiff.fr
}

def cleanup_fastdisk(pth):
    try:
        shutil.rmtree(pth)
        print('fast disk contents deleted.')
    except:
        print('fast disk location empty.')

def proccess_data(ops, db):
    print('Starting suite2p...')
    run_s2p(ops=ops, db=db)
    print('suite2p finished.')
    
if __name__ == '__main__':
    proccess_data(ops=default_ops, db=db)