import inspect
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sop
import scipy.stats as stats
from tqdm import tqdm

from ..stats import (gauss2d, mse, rsquared, sumsquares, von_mises,
                     von_mises_sym)
from ..utils import replace_tup_ix


def find_vis_resp(df, p=0.05, test='anova', vis_key='ori'):
    """
    Takes a mean dataframe (see meanby) and finds visually responsive cells using 
    a 1-way ANOVA test.
    
    Args:
        df (pd.DataFrame): mean dataframe (trials, cells, vis_condition)
        p (float, optional): p-valuse to use for significance. Defaults to 0.05.
        test (str, optional): statistical test to use, only one option now. Defaults to 'anova'.

    Returns:
        np.array of visually responsive cells
        np.array of p values for all cells
    """
    
    # for adding others later
    tests = {
        'anova': _vis_resp_anova(df, vis_key)
    }
    
    p_vals = tests[test]
    vis_cells = np.where(p_vals < p)[0]

    n = vis_cells.size
    c = p_vals.size
    print(f'There are {n} visually responsive cells, out of {c} ({n/c*100:.2f}%)')

    return vis_cells, p_vals

def po(mdf):
    """
    Takes a mean dataframe (see meanby) and returns preferred and
    orthagonal orientation in orientation space (mod 180).
    
    General procedure:
        1. Remove blank trial conditions (specified as -45 degs)
        2. Modulo 0-315* to 0-135* (mod excludes the number you put in)
        3. Get mean response by cell and orientation.
        4. Find index of max df, corresponding to PO.
        5. Subtract 90* from PO and mod 180 to get ortho

    Args:
        mdf (pd.DataFrame): mean response dataframe, generated from meanby (above)

    Returns:
        pd.Series of pref_oris
        pd.Series of ortho_oris
    """
    vals = mdf.loc[mdf.ori >= 0].copy()
    vals['ori'] = vals['ori'] % 180

    vals = vals.groupby(['cell', 'ori']).mean().reset_index()

    pref_oris = vals.set_index('ori').groupby('cell')['df'].idxmax()
    pref_oris.name = 'pref'
    
    ortho_oris = (pref_oris - 90) % 180
    ortho_oris.name = 'ortho'    

    return pref_oris, ortho_oris

def pdir(df):
    """Calculates pref dir."""
    df = df.loc[df.ori != -45]
    pref_dir = df.set_index('ori').groupby(['cell'])['df'].idxmax()
    pref_dir.name = 'pdir'

    return pref_dir

def osi(df):
    """
    Takes the mean df and calculates OSI.
    
    Procedure:
        1. Drop gray screen conditions (orientation == -45)
        2. Subtract off the minimum cell by cell. Note: it is VERY important do this
           to avoid negative values giving extremely high or low OSIs. Do this before
           averaging the tuning curves otherwise you get lots of OSIs = 1 (if ortho is
           is min and set to zero, OSI will always be 1).
        3. Groupby cell and ori to get mean dataframe/tuning curve.
        4. Get PO and OO values and calculate OSI.
    
    Returns a pd.Series of osi values
    
    Confirmed working by WH 7/30/20
    BUT LIKE REALLY REALLY FOR SURE THIS TIME
    
    """
    
    vals = df.loc[df.ori >= 0].copy()
    tc = vals.groupby(['cell', 'ori'], as_index=False).mean()
    
    osis = []
    for cell in vals.cell.unique():
        temp = tc[tc.cell == cell].copy()
        temp['df'] -= temp['df'].min()
        po = temp.df[temp.pref == temp.ori].values[0]
        o_deg1 = (temp.pref-90) % 360
        o_deg2 = (temp.pref+90) % 360
        o1 = temp.df[temp.ori == o_deg1].values[0]
        o2 = temp.df[temp.ori == o_deg2].values[0]
        oo = np.mean([o1, o2])
        osi = _osi(po,oo)
        osis.append(osi)

    osis = abs(np.array(osis))
    # osis = abs(osis[~np.isnan(osis)])

    osi = pd.Series(osi, name='osi')

    return osi

def _osi(preferred_responses, ortho_responses):
    """This is the hard-coded osi function."""
    return ((preferred_responses - ortho_responses)
            / (preferred_responses + ortho_responses))
    
def _global_osi(tuning_curve):
    # TODO
    pass

def _vis_resp_anova(data, vis_key):
    """Determine visual responsiveness by 1-way ANOVA."""

    f_val = np.empty(data.cell.nunique())
    p_val = np.empty(data.cell.nunique())

    for i,cell in enumerate(data.cell.unique()):
        temp3 = data[data.cell==cell]
        temp4 = temp3[[vis_key, 'trial', 'df']].set_index([vis_key,'trial'])
        samples = [col for col_name, col in temp4.groupby(vis_key)['df']]
        f_val[i], p_val[i] = stats.f_oneway(*samples)

    return p_val

def naka_rushton(c, a, b, c50, n):
    """
    Naka-Rushton equation for modeling contrast-response functions.
    
    Where: 
        c = contrast 
        a = Rmax (max firing rate)
        b = baseline firing rate
        c50 = contrast response at 50%
        n = exponent
    """
    return (a*(c**n)/((c50**n)+(c**n)) + b)

def naka_rushton_no_b(c, a, c50, n):
    """
    Naka-Rushton equation for modeling contrast-response functions. Do not fit baseline firing
    rate (b). Typical for pyramidal cells.
    
    Where: 
        c = contrast 
        a = Rmax (max firing rate)
        c50 = contrast response at 50%
        n = exponent
    """
    return (a*(c**n)/((c50**n)+(c**n)))
    
def naka_rushton_allow_decreasing(c, a, b, c50, n):
    """
    Naka-Rushton equation for modeling contrast-response functions. Taken from Dan to allow for 
    decreasing contrast responses (eg. VIP cells). If bounded, returns the same as naka_rushton. (?)
    
    Where: 
        c = contrast 
        a = Rmax (max firing rate)
        b = baseline firing rate
        c50 = contrast response at 50%
        n = exponent
    """
    return (a*(c/c50)**n + b)/((c/c50)**n + 1)

class NakaRushton:
    """More flexible model class for fitting Naka-Rushtons to CRF datasets."""
    def __init__(self) -> None:
        # fit options
        self.allow_decreasing = False
        self.fit_b = False
        self.constrain = True
        self._fitfun = None
        
        # guesses/bounds
        # self.guess = (max(response), 0, 25, 2)
        self.lower_bounds = (0, 0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, 100, 4)
        
        # results
        self.is_fit = False
        self.rmax = None
        self.b = None
        self.c50 = None
        self.n = None
        self.popt = None
        self.pcov = None
        
        # stats
        self.sse = None
        self.r2 = None
        
    @property
    def fitfun(self):
        if self._fitfun is None:
            if self.allow_decreasing:
                return naka_rushton_allow_decreasing
            else:
                return naka_rushton
        else:
            return self._fitfun
        
    def _check_bounds(self):
        # set bounds for b if not fitting to a very small number (avoids variable func calls)
        # move fit_b clause elsewhere for flexibility in creating general model classes
        if not self.fit_b:
            # set the bounds to not fit b (by constraining)
            self.lower_bounds = replace_tup_ix(self.lower_bounds, 1, 0)
            self.upper_bounds = replace_tup_ix(self.upper_bounds, 1, 0.000001)

            if not self.constrain:
                warnings.warn(f'Requested to not fit b, but constrain was set to False. Constraining and setting b_upper to {0.000001}.')
                self.constrain = True
        
        # make bounds array for easy comparisons        
        lb = np.array(self.lower_bounds)
        ub = np.array(self.upper_bounds)
        
        # ensure they are the same length
        if len(lb) != len(ub):
            raise ValueError('Length of lower and upper bounds must match.')
        
        # check length of fittable args is same as constraints
        args = inspect.getfullargspec(self.fitfun).args
        nvars = len(args[1:]) # first arg is X and not fit
        
        if len(lb) > nvars or len(ub) > nvars:
            raise ValueError(f'Number of bounds does not match number of fit function args ({nvars}).')
                
        # final checks and return clause, return nothing if not constrained
        if self.constrain:
            if np.any((lb < 0) | (ub < 0)):
                warnings.warn('Bounds set to negative. Are you sure?')
            if np.any(lb > ub):
                raise ValueError('Upper bounds may not be set lower than lower bounds.')
            return (self.lower_bounds, self.upper_bounds) 
        else:
            return
        
    def fit(self, contrast, response):
        bds = self._check_bounds()
        self.popt, self.pcov = sop.curve_fit(self.fitfun, contrast, response, bounds=bds)
        self.rmax, self.b, self.c50, self.n = self.popt
        self.is_fit = True
        
        # get SSE, R2, etc
        self.sse = sumsquares(contrast, response, self.fitfun, self.popt)
        self.r2 = rsquared(contrast, response, self.fitfun, self.popt)
    
    def predict(self, contrast):
        resp = self.fitfun(contrast, *self.popt)
        return resp
    
    def info(self):
        print(self.__repr__())
    
    def __repr__(self) -> str:
        if not self.is_fit:
            txt = f"""
            [Model]
            Naka-Rushton
            Fit fxn: <{self.fitfun.__module__}.{self.fitfun.__name__}>
            
            Not fit to any data (yet).
            """
        else:
            txt = f"""
            [Model]
            Naka-Rushton
            Fit fxn: <{self.fitfun.__module__}.{self.fitfun.__name__}>
            
            [Variables]
            Rmax: {self.rmax:.5f}
            b:    {self.b:.5f}
            c50:  {self.c50:.5f}
            n:    {self.n:.5f}
            
            [Statistics]
            SSE:  {self.sse:.5f}
            R2:   {self.r2:.5f}
            """
        return inspect.cleandoc(txt)
    
def get_ret_data(data: np.ndarray, win:tuple, locs:np.ndarray):
    """Proccess retinotopy PSTHs into responses by location.

    Args:
        data (np.ndarray): trialwise PSTHs
        win (tuple): response window to take mean responses IN FRAMES
        locs (np.ndarray): indices of xy locations of visual stimulus

    Returns:
        np.ndarray: cell x Y x X
    """
    data_ = data[:,:,win[0]:win[1]]
    Ny = locs[:,0].max()
    Nx = locs[:,1].max()
    ret = np.zeros((data_.shape[1], Ny, Nx))
    for j in range(Ny):
        for k in range(Nx):
            these_trials = (locs[:,0] == j+1) & (locs[:,1] == k+1)
            n_trials = these_trials.sum()
            for idx in np.where(these_trials)[0]:
                ret[:,j,k] = ret[:,j,k] + data_[idx,:,:].mean(1)/n_trials # what is going on here? summing across mean of trials?            
    return ret
    
class Retinotopy:
    def __init__(self, Nx:int, Ny:int, gridsize:int) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.gridsize = gridsize
        
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
            
        x_rng = np.linspace(-(self.Nx-1)*self.gridsize/2, (self.Nx-1)*self.gridsize/2, self.Nx*expand_by)
        y_rng = np.linspace(-(self.Ny-1)*self.gridsize/2, (self.Ny-1)*self.gridsize/2, self.Ny*expand_by)
        
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
    
def fit_all_ret(data, base_win, response_win, locinds, gridsize):
    
    Ny = locinds[:,0].max()
    Nx = locinds[:,1].max()

    ret = get_ret_data(data, response_win, locinds)
    
    ncells = ret.shape[0]
    ctr = np.full((ncells, 2), np.nan)
    pvals = np.full(ncells, np.nan)
    rvals = np.full(ncells, np.nan)
    fit_ret = []
    
    for i,r in enumerate(tqdm(ret, desc='Fitting retinotopy: ')):
        rfit = Retinotopy(Nx, Ny, gridsize)
        
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
    }
    time.sleep(2)
    
    return out

def find_center_cells(ctr_rfs, ctr_sz=7, ctr_fov=(0,0)):
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

class vonMises:
    def __init__(self) -> None:
        self.constrain = True
        self.bds = None
        self._fitfun = None
        
        self.is_fit = False
        self.popt = None
        self.pcov = None
        self.kappa = None
        self.mu = None
        self.ht = None
        self.po = None
        
    @property    
    def fitfun(self):
        if self._fitfun is None:
            return von_mises
        else:
            return self._fitfun
        
    @fitfun.setter
    def fitfun(self, func):
        self._fitfun = func
    
    def _get_bounds(self, data):
        if self.constrain == True:
            lb = (0, 0, 0)
            ub = (np.inf, np.inf, data.max()+1)
            bds = (lb, ub)
        else:
            lb = (-np.inf, -np.inf, -np.inf)
            ub = (np.inf, np.inf, np.inf)
            bds = (lb, ub)
        return bds
        
    def _check_radians(self, vals):
        if np.any(vals > np.deg2rad(360)):
            return np.deg2rad(vals)
        else:
            return vals
        
    def _check_nonneg(self, vals):
        if vals.min() < 0:
            vals -= vals.min()
        return vals
    
    def guess(self, oris: np.ndarray, data: np.ndarray):
        k = 5
        mu = oris[data.argmax()]
        ht = data.max()
        # print(k, mu, ht)
        return (k, mu, ht)
        
    def fit(self, oris: np.ndarray, responses: np.ndarray):
        oris_rad = self._check_radians(oris)
        responses = self._check_nonneg(responses)
        self.bds = self._get_bounds(responses)
        p0 = self.guess(oris, responses)
        self.popt, self.pcov = sop.curve_fit(self.fitfun, oris_rad, responses, p0=p0, bounds=self.bds)
        
        self.kappa, self.mu, self.ht = self.popt
        
        if oris.max() > 180:
            self.po_fit = np.rad2deg(self.mu) 
        else:
            self.po_fit = np.rad2deg(self.mu) % 180
        
        self.po = oris[self.predict(oris).argmax()]
        
        self.is_fit = True
        self.sse = sumsquares(oris, responses, self.fitfun, self.popt)
        self.r2 = rsquared(oris, responses, self.fitfun, self.popt)
        
    def predict(self, oris):
        oris_ = self._check_radians(oris)
        resp = self.fitfun(oris_, *self.popt)
        return resp
    
    def plot(self, oris=None, ax=None):
        if ax is None:
            ax = plt.gca()
            
        if oris is None:
            oris = np.arange(0,180)

        vals = self.predict(oris)
        ax.plot(vals)
        ax.set_xlabel('Orientation')
        ax.set_ylabel('Response')
    
    def info(self):
        print(self.__repr__())
    
    def __repr__(self) -> str:
        if not self.is_fit:
            txt = f"""
            [Model]
            Von Mises
            Fit fxn: <{self.fitfun.__module__}.{self.fitfun.__name__}>
            
            Not fit to any data (yet).
            """
        else:
            txt = f"""
            [Model]
            Von Mises
            Fit fxn: <{self.fitfun.__module__}.{self.fitfun.__name__}>
            
            [Variables]
            kappa:  {self.kappa:.5f}
            mu:     {self.mu:.5f}
            ht:     {self.ht:.5f}
            PO:     {self.po}
            PO_fit: {self.po_fit:.5f}
            
            [Statistics]
            SSE:    {self.sse:.5f}
            R2:     {self.r2:.5f}
            """
        return inspect.cleandoc(txt)