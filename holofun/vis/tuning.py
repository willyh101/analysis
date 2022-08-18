import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sop

from ..stats import rsquared, sumsquares, von_mises

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
