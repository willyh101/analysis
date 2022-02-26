import os
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List  # depricated in >py3.9
import json
import sys
from wsgiref.headers import tspecials

import numpy as np
import scipy
import tifffile
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        import caiman as cm
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction.cnmf import cnmf
        from caiman.source_extraction.cnmf.params import CNMFParams
        from caiman.source_extraction.cnmf.online_cnmf import OnACID
        
    warnings.simplefilter('ignore', category=ConvergenceWarning)
    warnings.simplefilter('ignore', category=DeprecationWarning)
    
except ModuleNotFoundError:
    pass

from .tiffs import SItiff, get_tslice, slice_movie, tiffs2array
from .utils import ptoc, ptoc_min, tic, toc


def transfer_movies(src_folders:List[Path], tmp_folder:Path, zplane:int, xtrim=0, channel=0):
    # delete contents of temp folder
    print('Deleting contents of caiman_temp...')
    for f in tmp_folder.iterdir():
        f.unlink()
        
    # detect movies and transfer to tmp folder
    mlist = []
    length_list = []
    src_movs = []
    files_per_epoch = []
    
    mlist_by_epoch = [sorted(src_folder.glob('*.tif*')) for src_folder in src_folders]
    
    # get basic tiff info for temporal slicing
    tiff = SItiff(mlist_by_epoch[0][0])
    xslice = slice(xtrim,512-xtrim)
    yslice = slice(None)
    tslice = get_tslice(zplane, channel, tiff.nchannels, tiff.nplanes)
    
    print('Slicing and moving movies...')
    for i,movs in enumerate(mlist_by_epoch):
        print(f'Transferring epoch {i}')
        f_count = 0 # have to do this hacky way to check file lengths are valid
        time.sleep(1)
        for m in tqdm(movs):
            temp_mov = slice_movie(str(m), xslice, yslice, tslice)
            # skip movie if it's too short
            if temp_mov.shape[0] < 10:
                continue
            length_list.append(temp_mov.shape[0])
            fname = f'temp_{i:05}.tif'
            tifffile.imsave(fname, temp_mov)
            mlist.append(fname)
            src_movs.append(str(m))
            f_count += 1
        files_per_epoch.append(f_count)
                
    return mlist, length_list, src_movs, files_per_epoch

def find_com(A, dims, x_1stPix):
    XYcoords= cm.base.rois.com(A, *dims)
    XYcoords[:,1] = XYcoords[:,1] + x_1stPix #add the dX from the cut FOV
    i = [1, 0]
    return XYcoords[:,i] #swap them

def find_com_man(A, dx, dy, x_1stPix):
    if not isinstance(A, scipy.sparse.csc_matrix):
        A = scipy.sparse.csc_matrix(A)
        
    Coor = np.matrix([np.outer(np.ones(dy), np.arange(dx)).ravel(),
                          np.outer(np.arange(dy), np.ones(dx)).ravel()],
                         dtype=A.dtype)
    com = (Coor * A / A.sum(axis=0)).T
    com = np.array(com)
    com[:,1] = com[:,1] + x_1stPix
    i = [1,0]
    return com[:,i]

def load_A_asarray(h):
    data =  h['estimates/A/data']
    indices = h['estimates/A/indices']
    indptr = h['estimates/A/indptr']
    shape = h['estimates/A/shape']
    A = scipy.sparse.csc_matrix((data[:], indices[:], indptr[:]), shape[:]).toarray()
    return A

def reshape_A(A, dims):
    return A.reshape((*dims, -1), order='F')

def run_caiman_batch(opts:dict, params:dict, plane:int):
    tt = tic()
    t_setup = tic()
    print(f'Starting plane {plane}...')
    print('Begin setting up and moving files.')
    
    
    ##---GENERAL SETUP---##
    results_root = opts['results_root']
    mouse = opts['mouse']
    date = opts['date']
    epochs = opts['epochs']
    
    epochs_folder_name = '_'.join(epochs)
    caiman_folder = Path(results_root, mouse, date, epochs_folder_name, 'caiman')
    caiman_folder.mkdir(exist_ok=True, parents=True)
    
    # change to temp folder
    os.chdir(opts['caiman_temp'])
    
    # move movies
    ch = opts.setdefault('channel', 0)
    xtrim = opts.setdefault('xtrim', 0)
    src = opts['src_folders']
    tmp = opts['caiman_temp']
    
    result = transfer_movies(src, tmp, plane, xtrim=xtrim, channel=ch)
    movs, file_lengths, src_fnames, files_per_epoch = result
    opts['frames_per_file'] = file_lengths
    opts['tiffs'] = src_fnames
    opts['files_per_epoch'] = files_per_epoch
    opts['plane'] = plane
    opts['date_ran'] = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    
    # make params object
    params['fnames'] = movs
    params = CNMFParams(params_dict=params)
    
    opts['t_setup'] = toc(t_setup)
    ptoc(t_setup, 'Setup complete.')
    ptoc_min(tt)
    
    
    ##---MOTION CORRECTION---##
    t_motion = tic()
    print('Starting caiman CNMF (batch) pipeline.')
    print('Begin motion correction.')
    
    # start cluster
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    
    # do motion correction
    mc = MotionCorrect(movs, dview=dview, **params.get_group('motion'))
    mc.motion_correct(save_movie=True)
    
    # memory map the file in order 'C'
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0 
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                        border_to_0=border_to_0, dview=dview) # exclude borders
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F') 
        #load frames in python format (T x X x Y)
    opts['mmap_file'] = fname_new
        
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    
    opts['t_motion'] =  toc(t_motion)
    ptoc(t_motion, 'Motion correction complete.')
    ptoc_min(tt)
    
    
    ##---CNMF---##
    t_cnmf = tic()
    print('Begin CNMF batch.')
    
    cnm = cnmf.CNMF(n_processes, params=params, dview=dview)
    cnm = cnm.fit(images)
    cnm.estimates.detrend_df_f()
    cnm.estimates.deconvolve(cnm.params)
    cnm.estimates.CoM = find_com(cnm.estimates.A, cnm.dims, 0)
    cnm.estimates.CoM_adj = find_com(cnm.estimates.A, cnm.dims, xtrim)
    
    # rerun CNMF on the full FOV
    cnm2 = cnm.refit(images, dview=dview)
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    cnm2.estimates.detrend_df_f()
    cnm2.estimates.deconvolve(cnm2.params)
    cnm2.estimates.CoM = find_com(cnm.estimates.A, cnm.dims, 0)
    cnm2.estimates.CoM_adj = find_com(cnm.estimates.A, cnm.dims, xtrim)

    # create and save images
    Cn = cm.local_correlations(images.transpose(1,2,0))
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.Cn = Cn
    cnm.estimates.mean_img = images.mean(0)
    cnm.estimates.max_img = images.max(0)
    cnm.estimates.med_img = np.median(images, 0)
    cnm2.estimates.Cn = Cn
    cnm2.estimates.mean_img = images.mean(0)
    cnm2.estimates.max_img = images.max(0)
    cnm2.estimates.med_img = np.median(images, 0)
    
    cm.stop_server(dview=dview)
    
    opts['t_cnmf'] = toc(t_cnmf)
    ptoc(t_cnmf, 'CNMF complete.')
    opts['t_total'] = toc(tt)
    ptoc_min(tt)
    
    for k,v in opts.items():
        setattr(cnm, k, v)
    
    for k,v in opts.items():
        setattr(cnm2, k, v)
    
    
    
    ##---SAVE--##
    out_pth = str(caiman_folder/'caiman_plane{plane}_init.hdf5')
    cnm.save(out_pth)
    print(f'Saved results to: {out_pth}')
    out_pth = str(caiman_folder/'caiman_plane{plane}.hdf5')
    cnm2.save(out_pth)
    print(f'Saved results to: {out_pth}')
    ptoc_min(tt, 'Plane completed.')


def run_caiman_onacid(opts:dict, params:dict, plane:int):
    # this could run more effiecently by streaming in the file from a remote source, but the fit_online
    # method really simplies running the algo
    tt = tic()
    t_setup = tic()
    print(f'Starting plane {plane}...')
    print('Begin setting up and moving files.')
    
    
    ##---GENERAL SETUP---##
    results_root = opts['results_root']
    mouse = opts['mouse']
    date = opts['date']
    epochs = opts['epochs']
    
    epochs_folder_name = '_'.join(epochs)
    caiman_folder = Path(results_root, mouse, date, epochs_folder_name, 'caiman')
    caiman_folder.mkdir(exist_ok=True, parents=True)
    
    # change to temp folder
    os.chdir(opts['caiman_temp'])
    
    # move movies
    ch = opts.setdefault('channel', 0)
    xtrim = opts.setdefault('xtrim', 0)
    src = opts['src_folders']
    tmp = opts['caiman_temp']
    
    result = transfer_movies(src, tmp, plane, xtrim=xtrim, channel=ch)
    movs, file_lengths, src_fnames, files_per_epoch = result
    opts['frames_per_file'] = file_lengths
    opts['tiffs'] = src_fnames
    opts['files_per_epoch'] = files_per_epoch
    opts['plane'] = plane
    opts['date_ran'] = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    
    # make params object
    params['fnames'] = movs
    params = CNMFParams(params_dict=params)
    
    opts['t_setup'] = toc(t_setup)
    ptoc(t_setup, 'Setup complete.')
    ptoc_min(tt)
    
    
    ##---OnACID---##
    t_acid = tic()
    print('Starting caiman OnACID pipeline.')
    cnm = OnACID(params=params)
    cnm.fit_online() # ENSURE MOTION CORRECT IS SET TO TRUE!!!
    cnm.estimates.detrend_df_f()
    cnm.estimates.evaluate_components_CNN(cnm.params)
    cnm.estimates.CoM = find_com(cnm.estimates.A, cnm.dims, 0)
    cnm.estimates.CoM_adj = find_com(cnm.estimates.A, cnm.dims, xtrim)
    
    
    opts['t_cnmf'] = toc(t_acid)
    ptoc(t_acid, 'CNMF complete.')
    opts['t_total'] = toc(tt)
    ptoc_min(tt)
    
    for k,v in opts.items():
        setattr(cnm, k, v)
    
    
    ##---SAVE--##
    out_pth = str(caiman_folder/'caiman_onacid_plane{plane}.hdf5')
    cnm.save(out_pth)
    print(f'Saved results to: {out_pth}')
    ptoc_min(tt, 'Plane completed.')
    
    
# def run_caiman_onacid_streaming(opts:dict, params:dict, plane:int):
#     tt = tic()
#     t_setup = tic()
#     print(f'Starting plane {plane}...')
#     print('Begin setting up and moving files.')
    
    
#     ##---GENERAL SETUP---##
#     results_root = opts['results_root']
#     mouse = opts['mouse']
#     date = opts['date']
#     epochs = opts['epochs']
    
#     epochs_folder_name = '_'.join(epochs)
#     caiman_folder = Path(results_root, mouse, date, epochs_folder_name, 'caiman')
#     caiman_folder.mkdir(exist_ok=True, parents=True)
    
#     # change to temp folder
#     os.chdir(opts['caiman_temp'])
    
#     # move movies
#     ch = opts.setdefault('channel', 0)
#     xtrim = opts.setdefault('xtrim', 0)
#     src = opts['src_folders']
#     tmp = opts['caiman_temp']
#     opts['plane'] = plane
    
    
# def append_to_queue_multifolder(q, tiff_folders, out_folder, tslice, add_rate=0.5):
#     # first, iterate through the epochs
#     files_per_epoch = []
#     lengths_list = []
#     for tiff_folder in tiff_folders:
#         tiff_list = Path(tiff_folder).glob('*.tif*')
#         f_count =  0
#         lengths = []
#         # then through files in each epoch
#         for i,t in enumerate(tiff_list):
#             # open data
#             with ScanImageTiffReader(str(t)) as reader:
#                 data = reader.data()
                
#             # check if valid tiff
#             if data.shape[0] > 15:    
#                 # slice movie for this plane
#                 mov = data[tslice, :, :]
#                 lengths.append(mov.shape[0])
#                 f_count += 1
#                 # add frames to the queue
#                 for f in mov:
#                     q.put(f.squeeze())         
#             else:
#                 continue   
#             # so we don't overload memory
#             time.sleep(add_rate)
        
#         # append the file count per epoch
#         files_per_epoch.append(f_count)
#         lengths_list.append(lengths)
                
#     fname = Path(out_folder,'file_lengths.json')
#     data = dict(lengths=lengths_list, files_per_epoch=files_per_epoch)
#     with open(fname, 'w') as f:
#         json.dump(data, f)
        
#     q.put('STOP')
    
# def prepare_init(plane: int, n_init: int, tiff_files: list):
#     nframes = 0
#     init_list = []
#     print('getting files for initialization....')
#     while nframes < n_init:
#         tiff = next(tiff_files)
#         if nframes == 0:
#             im = SItiff(tiff)
#             nchannels = im.nchannels
#             nplanes = im.nplanes
#             tslice = get_tslice(plane, 0, nchannels, nplanes)
#         length = slice_movie(str(tiff), slice(None), slice(None), tslice).shape[0]
#         if length > 10:
#             init_list.append(tiff)
#             nframes += length
#         else:
#             continue
#     return init_list, nchannels, nplanes, tslice

# def process_streaming(opts:dict, params:CNMFParams):
#     # assumes the working directory is already tmp
#     plane = opts['plane']
#     n_init = opts.get('n_init', 500)
#     files_init, nchannels, nplanes, tslice = prepare_init(plane, n_init, tiff_files)
#     mov = tiffs2array(files_init, )