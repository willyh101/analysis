import pandas as pd

def make_holo_df(hr_targets):
    holos = pd.DataFrame({
        'hr targets': hr_targets,
    })
    
    # add holo size lengths
    vals = holos['hr targets'].apply(len).values
    holos['num targets requested'] = vals
    
    return holos

class Holo:
    def __init__(self, targs) -> None:
        self.targs = targs
        
        
        self.ntargeted = len(targs)
        self.nmatched = None
        self.off_target_range = 8 # microns
        self.off_targets = None
        self.xyz = None
        
