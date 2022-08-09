import inspect
import warnings

import numpy as np
import scipy.optimize as sop

from ..stats import rsquared, sumsquares
from ..utils import replace_tup_ix


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
