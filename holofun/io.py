import numpy as np
import h5py
import scipy.io as sio

def load_fg_result(path):
    with h5py.File(path) as h:
        out = {
            'dirs': h['result/dirs'][:].squeeze().astype(int),
            'stims': h['result/stims'][:].squeeze().astype(int),
            'stim_key': ['large', 'cross', 'center', 'hole'],
            'sf': float(h['result/sf'][:]),
            'tf': float(h['result/tf'][:]),
            'fsize': int(h['result/fsize'][:]),
            'gsize': int(h['result/gsize'][:])
        }
    return out

def load_ori_data(path):
    mat = sio.loadmat(path, squeeze_me=True)['result']
    stims = mat['stimParams'].item()
    out = {
        'ori': stims[0,:],
        'size': stims[1,:],
        'contrast': stims[4,:]
    }
    out['ori'][np.isnan(out['ori'])] = -1
    return out

def load_ret_data(path):
    result = sio.loadmat(path, squeeze_me=True)['result'][()]
    locinds = result['locinds']
    out = {
        'locinds': locinds,
        'gridsize': result['sizes'],
        'Ny': locinds[:,0].max(),
        'Nx': locinds[:,1].max(),
        'gridsample': result['grid']
    }
    return out