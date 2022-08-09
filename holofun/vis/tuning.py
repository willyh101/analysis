import inspect

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sop

from ..stats import rsquared, sumsquares, von_mises


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
