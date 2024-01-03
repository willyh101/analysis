import inspect
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sop
import scipy.stats as stats
from tqdm import tqdm

from ..stats import gauss2d, mse, rsquared, sumsquares


def get_ret_data(data: np.ndarray, win:tuple, locs:np.ndarray):
    """Proccess retinotopy PSTHs into responses by location.

    Args:
        data (np.ndarray): trialwise PSTHs
        win (tuple): response window to take mean responses IN FRAMES
        locs (np.ndarray): indices of xy locations of visual stimulus

    Returns:
        np.ndarray: cell x Y x X
    """
    if len(win) == 2:
        # assumes win represents response window in frames
        data_ = data[:,:,win[0]:win[1]]
    elif len(win) == 4:
        # assunes win has baseline and response window in frames
        data_ = data[:,:,win[2]:win[3]] - data[:,:,win[0]:win[1]].mean(2, keepdims=True)
    else:
        raise ValueError('win must be a tuple of length 2 or 4')
    Ny = locs[:,0].max()
    Nx = locs[:,1].max()
    ret = np.zeros((data_.shape[1], Ny, Nx))
    
    if data_.shape[0] > locs.shape[0]:
        logging.warning('Ret locs less than number of trials collected! Auto-adjusting...')
        locs = np.tile(locs, (30,1))
        locs = locs[:data_.shape[0],:]
    
    for j in range(Ny):
        for k in range(Nx):
            these_trials = (locs[:,0] == j+1) & (locs[:,1] == k+1)
            n_trials = these_trials.sum()
            for idx in np.where(these_trials)[0]:
                ret[:,j,k] = ret[:,j,k] + data_[idx,:,:].mean(1)/n_trials # what is going on here? summing across mean of trials?            
    return ret
    
class Retinotopy:
    def __init__(self, Nx:int, Ny:int, gridsize:int, gridsample=1) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.gridsize = gridsize
        self.gridsample = gridsample
        
        self._fitfun = None
        self.bds = None
        
        # results
        self.is_fit = False
        self.popt = None
        self.pcov = None
        self.xo = None
        self.yo = None
        self.a = None
        self.sigma_x = None
        self.sigma_y = None
        self.theta = None
        self.offset = None
    
    @property
    def fitfun(self):
        if self._fitfun is None:
            return gauss2d
        else:
            return self._fitfun
        
    def calculate_grid(self, expand_by=1):
            
        x_rng = np.linspace(-(self.Nx-1)*self.gridsize/2/self.gridsample, 
                            (self.Nx-1)*self.gridsize/2/self.gridsample, self.Nx*expand_by)
        y_rng = np.linspace(-(self.Ny-1)*self.gridsize/2/self.gridsample, 
                            (self.Ny-1)*self.gridsize/2/self.gridsample, self.Ny*expand_by)
        
        xx,yy = np.meshgrid(x_rng, y_rng)
        
        x = xx.flatten()
        y = yy.flatten()
        sz = xx.shape
        
        return  (x, y, sz)
    
    def get_bounds(self):
        x, y, sz = self.calculate_grid()
        lb = (x.min(), y.min(), 0, 0, 0, 0, 0)
        ub = (x.max(), y.max(), np.inf, x.max(), y.max(), 2*np.pi, np.inf)
        ub = (x.max(), y.max(), np.inf, 20, 20, 2*np.pi, np.inf)
        self.bds = (lb, ub)
        
    def guess(self, ret):
        x, y, sz = self.calculate_grid()
        extremum = np.argmax(ret)
        p0 = (x[extremum], y[extremum], ret.max()-ret.min(), 10, 10, 0, np.maximum(0, ret.min()))
        return p0
            
    def fit(self, ret: np.ndarray):
        x, y, sz = self.calculate_grid()
        self.get_bounds()
        p0 = self.guess(ret)
        
        # fit
        self.popt, self.pcov = sop.curve_fit(gauss2d, (x,y), ret.ravel(), p0=p0, bounds=self.bds)
        self.xo, self.yo, self.a, self.sigma_x, self.sigma_y, self.theta, self.offset = self.popt
        self.is_fit = True
        
        # score
        self.sse = sumsquares((x,y), ret.ravel(), self.fitfun, self.popt)
        self.mse = mse((x, y), ret.ravel(), self.fitfun, self.popt)
        self.r2 = rsquared((x,y), ret.ravel(), self.fitfun, self.popt)
        
    def predict(self, expand_by=1):
        x, y, sz = self.calculate_grid(expand_by)
        resp = self.fitfun((x,y), *self.popt)
        return resp.reshape(sz)

    def plot(self, expand_by=1, ax=None):
        if ax is None:
            ax = plt.gca()

        x, y, _ = self.calculate_grid(expand_by)
        ret = self.predict(expand_by)
        ext = (x.min(), x.max(), y.min(), y.max())
        ax.imshow(ret, extent=ext)
        
    def info(self):
        print(self.__repr__())
    
    def __repr__(self) -> str:
        if not self.is_fit:
            txt = f"""
            [Model]
            2D Gaussian Retinotopy
            Fit fxn: <{self.fitfun.__module__}.{self.fitfun.__name__}>
            
            Not fit to any data (yet).
            """
        else:
            txt = f"""
            [Model]
            2D Gaussian Retinotopy
            Fit fxn: <{self.fitfun.__module__}.{self.fitfun.__name__}>
            
            [Variables]
            Xo:      {self.xo:.5f}
            Yo:      {self.yo:.5f}
            amp:     {self.a:.5f}
            sigma_x: {self.sigma_x:.5f}
            sigma_y: {self.sigma_y:.5f}
            theta:   {self.theta:.5f}
            offset:  {self.offset:.5f}
            
            [Statistics]
            SSE:     {self.sse:.5f}
            MSE:     {self.mse:.5f}
            R2:      {self.r2:.5f}
            """
        return inspect.cleandoc(txt)
    
def fit_all_ret(data: np.ndarray, base_win: tuple[int,int], response_win: tuple[int,int], 
                locinds: np.ndarray, gridsize: int, baseline_resp = False, **kwargs) -> dict:
    
    Ny = locinds[:,0].max()
    Nx = locinds[:,1].max()

    if baseline_resp:
        win = (*base_win, *response_win)
    else:
        win = response_win

    ret = get_ret_data(data, win, locinds)
    
    ncells = ret.shape[0]
    ctr = np.full((ncells, 2), np.nan)
    pvals = np.full(ncells, np.nan)
    rvals = np.full(ncells, np.nan)
    fit_ret = []
    
    for i,r in enumerate(tqdm(ret, desc='Fitting retinotopy: ')):
        rfit = Retinotopy(Nx, Ny, gridsize, **kwargs)
        
        try:
            rfit.fit(r)
            ff = rfit.predict()
            fit_ret.append(ff)
            ctr[i,:] = [rfit.xo, rfit.yo]
            _,p = stats.ttest_rel(data[:,i,base_win[0]:base_win[1]].mean(1), 
                                  data[:,i,response_win[0]:response_win[1]].mean(1))
            pvals[i] = p
            rvals[i] = rfit.r2
            
        except RuntimeError:
            fit_ret.append(np.nan)
            
    out = {
        'ctr': ctr,
        'pvals': pvals,
        'rvals': rvals,
        'fits': fit_ret,
        'mean_by_loc': ret
    }
    time.sleep(2)
    
    return out

def find_center_cells(ctr_rfs: np.ndarray, ctr_sz=7, ctr_fov=(0,0)) -> np.ndarray:
    """
    Finds cells aligned to the retinotopic center. Returns a boolean of center aligned cells.

    Args:
        ctr_rfs (np.ndarray): xy coordinates of fit receptive field centers. (cell x XY)
        ctr_sz (int, optional): size of the center you want to select for. Defaults to 7.
        ctr_fov (tuple, optional): location of the center in visual degrees. Defaults to (0,0).
    """
    
    # calculate distance from ctr_fov
    dist = np.linalg.norm(ctr_rfs - np.array(ctr_fov), axis=1)
    ctr_cells = dist < ctr_sz
    
    return ctr_cells

def find_center_cells_box(ctr_rfs, ctr_sz=7, ctr_fov=(0,0)):
    """
    Finds cells aligned to the retinotopic center. Returns a boolean of center aligned cells.

    Args:
        ctr_rfs (np.ndarray): xy coordinates of fit receptive field centers. (cell x XY)
        ctr_sz (tuple or int, optional): size of the center you want to select for. Defaults to 7.
        ctr_fov (tuple, optional): location of the center in visual degrees. Defaults to (0,0).
    """
    
    if isinstance(ctr_sz, int):
        ctr_sz = (ctr_sz, ctr_sz)
    
    # find bbox
    x = np.arange(ctr_fov[0]-ctr_sz[0], ctr_fov[0]+ctr_sz[0]+1)
    y = np.arange(ctr_fov[1]-ctr_sz[0], ctr_fov[1]+ctr_sz[1]+1)
    ll = np.array([x.min(), y.min()])
    ur = np.array([x.max(), y.max()])
    
    ctr_cells = np.all((ll <= ctr_rfs) & (ctr_rfs <= ur), axis=1)
    
    return ctr_cells
