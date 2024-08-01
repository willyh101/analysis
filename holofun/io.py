import h5py
import numpy as np
import scipy.io as sio

from .utils import failsgraceful


def load_multi_vis(path):
    with h5py.File(path) as h:
        stim_data = {
            'dirs': h['result/dirs'][:].squeeze().astype(int),
            'stims': h['result/stims'][:].squeeze().astype(int)
        }
        meta1 = {
            'sf': float(h['result/sf'][:]),
            'tf': float(h['result/tf'][:]),
            'fsize': int(h['result/fsize'][:]),
            'gsize': int(h['result/gsize'][:]),
            'stim_ids': ['center', 'center+iso', 'center+cross', 'control'],
            'duration': int(h['result/duration'][:]),
        }
        try:
            meta2 = {
                'contrast_diff': float(h['result/contrast_diff'][:]),
                'contrast_diff_outer': float(h['result/contrast_diff_outer'][:]),
            }
        except KeyError:
            meta2 = {
                'contrast_diff': 'MISSING',
                'contrast_diff_outer': 'MISSING'
            }
        meta = {**meta1, **meta2}
        out = {'stims': stim_data, 'meta': meta}
    return out

def load_surr_stim_data(path):
    with h5py.File(path) as h:
        stim_data = {
            'dirs': h['result/dirs'][:].squeeze().astype(int),
            'stims': h['result/stims'][:].squeeze().astype(int)
        }
        meta1 = {
            'sf': float(h['result/sf'][:]),
            'tf': float(h['result/tf'][:]),
            'fsize': int(h['result/fsize'][:]),
            'gsize': int(h['result/gsize'][:]),
            'stim_ids': ['center', 'center+iso', 'control'],
            'duration': int(h['result/duration'][:]),
        }
        try:
            meta2 = {
                'contrast_diff': float(h['result/contrast_diff'][:]),
                'contrast_diff_outer': float(h['result/contrast_diff_outer'][:]),
            }
        except KeyError:
            meta2 = {
                'contrast_diff': 'MISSING',
                'contrast_diff_outer': 'MISSING'
            }
        meta = {**meta1, **meta2}
        out = {'stims': stim_data, 'meta': meta}
    return out


@failsgraceful
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

@failsgraceful
def load_ori_data(path):
    mat = sio.loadmat(path, squeeze_me=True)['result']
    stims = mat['stimParams'].item()
    out = {
        'ori': stims[0,:],
        'sz': stims[1,:],
        'size': stims[1,:],
        'contrast': stims[4,:]
    }
    out['ori'][np.isnan(out['ori'])] = -1
    return out

@failsgraceful
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