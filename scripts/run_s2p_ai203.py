from suite2p.run_s2p import run_s2p
import os
import shutil
import glob
from ScanImageTiffReader import ScanImageTiffReader
import sys

# data location and inputs
# animalid = 'w32_2'
# date = '20210420'
# expt_ids = ['1','2','3']
# result_base = 'E:/functional connectivity'

animalid = 'w20_1'
date = '20191230'
expt_ids = ['stimtest','ori1','ori2', 'expt']
result_base = 'E:/ai203/new_s2p'

diameter = 8

default_ops = {
    # general
    'diameter': diameter,
    'fast_disk': 'E:/s2p_scratch',
    'do_bidiphase': True,
    'save_mat': True,
    'tau': 1,
    'delete_bin': False, # true
    
    # registration
    'keep_movie_raw': True,
    'two_step_registration': False,
    'nimg_init': 1000, # subsampled frames for finding reference image
    'batch_size': 2000, # number of frames per batch, default=500
    
    # non rigid registration settings
    'nonrigid': False, # whether to use nonrigid registration
    
    # cell extraction
    'roidetect': True,
    'denoise': False,
    'threshold_scaling': 0.6, # was 1, adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
    'sparse_mode': False,
    'spatial_scale': 0,  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
    
    # custom settings
    'remove_artifacts': (120,512-120)
}

def make_db(animalid, date, expt_ids, result_base, raw_base='D:/Frankenrig/Experiments/'):
    save_path0 = os.path.join(result_base, animalid, date, '_'.join(expt_ids))
    data_path = [os.path.join(raw_base, animalid, date, lbl) for lbl in expt_ids]
    
    with ScanImageTiffReader(glob.glob(data_path[0]+'/*.tif')[0]) as reader:
        metadata = reader.metadata()
        
    db = {
        'save_path0': save_path0,
        'data_path': data_path,
        'nchannels': 1,#len(metadata.split('channelSave = [')[1].split(']')[0].split(';')),
        'nplanes': len(metadata.split('hStackManager.zs = [')[1].split(']')[0].split(' ')),
        'fs': float(metadata.split('scanVolumeRate = ')[1].split('\n')[0])
    }
    
    return db
    
def process_data(ops, db):
    fast_disk = ops['fast_disk'] + '/suite2p'
    try:
        shutil.rmtree(fast_disk + '/suite2p')
        print('fast disk contents deleted.')
    except:
        print('fast disk location empty.')

    print('Starting suite2p...')
    
    run_s2p(ops=ops,db=db)


if __name__ == '__main__':
    db = make_db(animalid, date, expt_ids, result_base)
    process_data(ops=default_ops, db=db)