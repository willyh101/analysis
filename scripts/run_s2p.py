from suite2p.run_s2p import run_s2p
import os
import shutil
import glob
from ScanImageTiffReader import ScanImageTiffReader

# data location and inputs
# animalid = 'w32_2'
# date = '20210420'
# expt_ids = ['1','2','3']
# result_base = 'E:/functional connectivity'

animalid = 'w42_2'
date = '20220216'
expt_ids = ['1', '2', '3', '5']


result_base = 'E:/functional connectivity'
tiff_base = 'D:/Frankenrig/Experiments/'

diameter = 8

default_ops = {
    # general
    'diameter': diameter,
    'fast_disk': 'K:/tmp/s2p_python',
    'do_bidiphase': True,
    'save_mat': False,
    'tau': 1.0,
    
    # registration
    'keep_movie_raw': False, # must be true for 2 step reg
    'two_step_registration': False,
    'nimg_init': 500, # subsampled frames for finding reference image
    'batch_size': 500, #2000, # number of frames per batch, default=500
    
    # non rigid registration settings
    'nonrigid': False, # whether to use nonrigid registration
    
    # cell extraction
    'denoise': False,
    'threshold_scaling': 0.8, # adjust the automatically determined threshold by this scalar multiplier, was 1. (WH) # 0.6 for low signal, default 5
    'sparse_mode': False,
    'max_iterations': 50, # usualy stops at threshold scaling, default 20
    'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P), default 100
    
    # custom settings
    'remove_artifacts': (105,512-105)
}

def make_db(animalid, date, expt_ids, result_base, tiff_base):
    save_path0 = os.path.join(result_base, animalid, date, '_'.join(expt_ids))
    data_path = [os.path.join(tiff_base, animalid, date, lbl) for lbl in expt_ids]
    
    with ScanImageTiffReader(glob.glob(data_path[0]+'/*.tif')[0]) as reader:
        metadata = reader.metadata()
        
    db = {
        'save_path0': save_path0,
        'data_path': data_path,
        'nchannels': len(metadata.split('channelSave = [')[1].split(']')[0].split(';')),
        'nplanes': len(metadata.split('hStackManager.zs = [')[1].split(']')[0].split(' ')),
        # 'nplanes': 1,
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
    
    print('emptying contents of temp fast disk folder...', end=' ')
    
    try:
        shutil.rmtree(fast_disk + '/sute2p')
        print('done.')
        
    except:
        print('failed to clean up fast disk!')
        
    print('suite2p finished.')


if __name__ == '__main__':
    db = make_db(animalid, date, expt_ids, result_base, tiff_base)
    process_data(ops=default_ops, db=db)