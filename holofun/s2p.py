import numpy as np
from pathlib import Path

from .si_tiff import get_crop_mask
from .simple_guis import openfoldergui

def eucl_motion(x_off, y_off):
    """
    Calculates the euclidian distance of motion on a trialwise basis using X and Y offsets.
    """
    offsets = np.vstack([x_off, y_off])
    offsets = offsets - np.median(offsets, axis=1).reshape(-1,1)
    return np.linalg.norm(offsets, axis=0)

def neuropil_subtract(F_raw, Neu, np_coeff):
    return F_raw - np_coeff*Neu

class Suite2pData:
    """
    Class holding suite2p data and methods for analysis.
    """
    
    def __init__(self, s2p_path: str) -> None:
        """
        Loads the suite2p data from file.

        Args:
            s2p_path (str): path to suite2p folder.
        """
        self.path = Path(s2p_path)
        
        if self.path.name != 'suite2p':
            raise NameError('You must select the root suite2p folder.')
        
        # file locations, stat, and ops files
        self.plane_folders = list(self.path.glob('plane*'))
        self.epoch_names = self.path.parent.name.split('_')
        self.ops = [np.load(f/'ops.npy', allow_pickle=True).item() for f in self.plane_folders]
        self.stats = [np.load(f/'stat.npy', allow_pickle=True) for f in self.plane_folders]
        
        # basic experiment info
        self.nplanes = len(self.plane_folders)
        self.nchannels = self.ops[0]['nchannels']
        self.fr = self.ops[0]['fs']
        
        # ID clicked cells
        self.iscell = self.get_iscell()
        self.meds = self.get_iscell_meds()
        
        # get stat for clicked cells
        self.stat = self.get_stat_iscell()
        
        # get traces and do NP subtraction
        self.neucoeff = self.ops[0]['neucoeff']
        
        self.F_raw = self.load_traces_npy('F.npy')
        try:
            self.Neu = self.load_traces_npy('FNeu.npy')
        except FileNotFoundError:
            self.Neu = self.load_traces_npy('Fneu.npy')
        self.spks = self.load_traces_npy('spks.npy')
        
        self.F = neuropil_subtract(self.F_raw, self.Neu, self.neucoeff)
        
    def get_motion(self, z=0):
        return eucl_motion(self.ops[z]['xoff'], self.ops[z]['yoff'])
        
    def get_stat_iscell(self):
        stat_combined = np.concatenate(self.stats)
        iscell = self.iscell[:,0].astype(bool)
        return stat_combined[iscell]
        
    def get_iscell_meds(self, optotune_depths=None):
        data = []
        
        if optotune_depths:
            plane_zs = optotune_depths
        else:
            plane_zs = list(range(self.nplanes))
            
        for i, stat in enumerate(self.stats):
            info = np.array([cell_info['med'] for cell_info in stat]).astype(int)
            plane_z = plane_zs[i]
            z_vals = np.full((info.shape[0], 1), fill_value=plane_z)
            info = np.hstack((info, z_vals))            
            data.append(info)
            
        data = np.vstack(data)
        iscell = self.iscell[:,0].astype(bool)
        data[:] = data[:,[1,0,2]] # XY needs to be swapped for meds
        
        offset = self.ops[0].get('remove_artifacts', [0])[0] # index 0 bc we just want x start
        data[:,0] += offset
        
        return data[iscell]
           
    def load_npys(self, filename: str):
        """Get the specified file for all planes. Outputs a list."""
        if 'npy' not in filename.split('.'):
            filename = filename + '.npy'
        return [np.load(f/filename) for f in self.plane_folders]
    
    def load_traces_npy(self, npy_file: str, iscell=True):
        temp = self.load_npys(npy_file)
        sz = np.array([t.shape[1] for t in temp])
        if np.any(sz > min(sz)):
            temp_fixed = [t[:,:min(sz)] for t in temp]
            data = np.vstack(temp_fixed)
            print('WARNING! HACKY FIX FOR PLANES NOT HAVING SAME NUMBER OF POINTS IN TRACES!')
        else:
            data = np.vstack(temp)
        if iscell:
            data = data[self.iscell[:,0].astype('bool')]
        return data
        
    def get_iscell(self):
        iscell = self.load_npys('iscell.npy')
        return np.concatenate(iscell)
    
    def get_iscell_combined(self):
        """If getting iscell from combined need to recalucate everything."""
        # wonky way to do this whole thing, but whatever
        self.iscell = np.load(self.path/'combined/iscell.npy')
        
        # recalculate traces data
        files = ['F', 'FNeu', 'spks']
        attrs = ['F_raw', 'Neu', 'spks']
        
        for f,a in zip(files, attrs):
            setattr(self, a, self.load_traces_npy(f+'.npy'))
        
        # re-do npil subtraction
        self.F = neuropil_subtract(self.F_raw, self.Neu, self.neucoeff)
        
        # update meds
        self.meds = self.get_iscell_meds()
    
    def cut_traces_epoch(self, epoch: int, src='F') -> np.ndarray:
        """Get s2p traces for a specific epoch."""
        traces = getattr(self, src)
        traces_cut = np.split(traces, np.cumsum(self.ops[0]['frames_per_folder']), axis=1)
        return traces_cut[epoch]
    
    def get_epoch_trial_lengths(self, epoch):
        ops = self.ops[0]
        tiff_splits = np.where(ops['first_tiffs'])[0][1:] # don't include the first bc it's zero
        epoch_file_lengths = np.split(ops['frames_per_file'], tiff_splits)
        return epoch_file_lengths[epoch]
    
    def get_img(self, plane, ch=0):
        if ch == 0:
            return self.ops[plane]['meanImg']
        else:
            return self.ops[plane]['meanImg_chan2']
    
    def get_cell_mask(self, cell, bb_crop=None, maskval='lam'):
        stat = self.stat[cell]
        ypix = stat['ypix']#[~stat['overlap']]
        xpix = stat['xpix']#[~stat['overlap']]
        
        im = np.zeros((self.ops[0]['Ly'], self.ops[0]['Lx']))
        if isinstance(maskval, int):
            im[ypix,xpix] = maskval
        else:
            im[ypix,xpix] = stat['lam']
        
        if bb_crop:
            x, y = self.meds[cell,:2]
            x -= self.ops[0].get('remove_artifacts', [0])[0]
            crop_mask = get_crop_mask(x, y, bb_crop)
            im = im[crop_mask]
        
        return im
    
    def list_epochs(self):
        for i, ep in enumerate(self.epoch_names):
            print(f'{i}: {ep}')
        
    @classmethod
    def from_names(cls, result_folder, mouse, date, epoch='*'):
        pass
    
    @classmethod
    def select_file(cls, rootdir='e:/'):
        path = openfoldergui(rootdir=rootdir, title='Select suite2p Folder')
        if not path:
            return
        print(f'Opened s2p folder: {path}')
        # this calls the init
        return cls(path)