import inspect
import warnings

import numpy as np


class HoloOptimizer:
    def __init__(self) -> None:
        self.score_funcs = None
        self.update_func = None


class GenericModel:
    def __init__(self) -> None:
        pass
    
    def _check_bounds(self):
        # NOTE: this isn't going to work w/o lower_bounds, upper_bounds, and fitfun being defined somehow
        # TODO: how to get number of bounds and args dynamically?
        
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