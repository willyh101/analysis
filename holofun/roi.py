import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import scipy.ndimage as ndi

def get_med(mask: np.ndarray):
    """Return (Y,X) median location from binary mask."""
    ypix, xpix = mask.nonzero()
    med = np.array([np.median(ypix), np.median(xpix)], dtype=int)
    return med

def get_com(mask: np.ndarray):
    """Return (Y,X) center of mass location from binary mask."""
    return ndi.center_of_mass(mask)

def cellpose_masks_to_indiv(cp_masks: np.ndarray):
    arr = []
    for i in range(cp_masks.max()):
        arr.append(cp_masks==i)
    return np.array(arr)

class ROI:
    def __init__(self, mouse, date, idx) -> None:
        self.mouse = mouse
        self.date = date
        self.idx = idx
        self._uid = self._gen_human_readable_uid(mouse, date, idx)
        
        self.xpix = None
        self.ypix = None
        
        self.lam = None
        
        self._mask = None
        self.xy_shape = (512, 512)
        
    def mask_as_coo_matrix(self):
        mask = coo_matrix((self.lam, (self.ypix, self.xpix)), shape=self.xy_shape)
        return mask
    
    def mask_as_binary(self):
        mask = np.zeros(self.xy_shape)
        if self.lam is None:
            mask[self.ypix, self.xpix] = 1
        else:
            mask[self.ypix, self.xpix] = self.lam
        return mask
    
    def crop_mask(self, asarray=False):
        mask = self.mask_as_coo_matrix()
        rmin = mask.row.min()
        cmin = mask.col.min()
        rmax = mask.row.max()+1
        cmax = mask.col.max()+1
        crop_mask = mask.tocsr()[rmin:rmax, cmin:cmax].tocoo()
        if asarray:
            crop_mask = crop_mask.toarray()
        return crop_mask
        
    @property
    def mask(self):
        return self._mask
    
    @property
    def uid(self):
        return self._uid
    
    def _gen_hex_uid(self, *args):
        out = []
        for s in args:
            out.append(hex(s)[2:])
        return '_'.join(out)
    
    def _gen_human_readable_uid(self, *args):
        return '_'.join(args)
    
class Cells:
    def __init__(self, ncell):
        self.ncell = ncell
        self.idxs = np.arange(self.ncell)
        self.data = pd.DataFrame(index=self.idxs)
        self.trace = None
        
    @property
    def labels(self):
        return list(self.data.columns)
    
    def __getitem__(self, attr: str):
        if attr in self.labels:
            return self.data[attr]
        else:
            raise KeyError(f'Label `{attr}` is not a cell label.')
        
    def add_labels(self, **kwargs):
        for k,v in kwargs.items():
            assert len(v) == self.ncell
            self.data[k] = v
            
    def add_roi_uids(self, mouse, date):
        self.mouse = mouse
        self.date = date
        s = ['_'.join([mouse, date, str(i)]) for i in range(self.ncell)]
        self.data.loc[:, 'uid'] = s
        
    def find(self, cond):
        return np.where(cond)[0]
    
class Traces:
    def __init__(self, arr: np.ndarray):
        self.C = 0
        self.T = 1
        self.data = arr
        self.cells = Cells(ncell=self.data.shape[self.C])
        self.mouse = None
        self.date = None
        self.fs = None
        self.timebase_type = 'frame'
        self.time_frames = np.arange(self.nframes, dtype=int)
        self.time_secs = None
        
    @property
    def ncells(self):
        return self.shape[self.C]
    
    @property
    def nframes(self):
        return self.shape[self.T]
    
    @property
    def dims(self):
        return ('cell', self.timebase_type)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def time(self):
        if self.timebase_type == 'frame':
            return self.time_frames
        elif self.timebase_type == 'secs':
            return self.time_secs
    
    def convert_timebase_secs(self, fs):
        self.fs = fs
        self.time_secs = self.time_frames/self.fs
        self.timebase_type = 'secs'
        
    def reset_timebase(self):
        self.timebase_type = 'frame'
        self.fs = None
        self.time_secs = None
        
    def to_psths(self, trial_lengths):
        pass
            
    def exp_info(self):
        print('mouse:', self.mouse) 
        print('date:', self.date)
        print('dims:', self.shape, self.dims)
        
class PSTHs:
    def __init__(self, traces: Traces) -> None:
        pass

