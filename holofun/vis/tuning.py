import numpy as np
import pandas as pd

from .vonmises import vonMises


def mean_responses_to_mdf(mresp: np.ndarray):
    pass

def po(mdf: pd.DataFrame):
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
    vals = get_tc(mdf, ori=True)

    pref_oris = vals.set_index('ori').groupby('cell')['df'].idxmax()
    pref_oris.name = 'pref'
    
    ortho_oris = (pref_oris - 90) % 180
    ortho_oris.name = 'ortho'    

    return pref_oris, ortho_oris

def pdir(df: pd.DataFrame):
    """Calculates pref dir."""
    df = get_tc(df)
    pref_dir = df.set_index('ori').groupby(['cell'])['df'].idxmax()
    pref_dir.name = 'pdir'

    return pref_dir

def get_tc(mdf: pd.DataFrame, drop_grey_screen=True, ori=False, return_err=False):
    if drop_grey_screen:
        vals = mdf.loc[mdf.ori >= 0].copy()
    else:
        vals = mdf.copy()
        
    if ori:
        vals['ori'] = vals['ori'] % 180
        
    m = vals.groupby(['cell', 'ori'], as_index=False).mean()
    e = vals.groupby(['cell', 'ori'], as_index=False).sem()
    
    if return_err:
        return m, e
    else:
        return m
        
def dsi(mdf: pd.DataFrame):    
    
    tc = get_tc(mdf, drop_grey_screen=True, ori=False)
    
    osis = []
    for cell in tc.cell.unique():
        temp = tc[tc.cell == cell].copy()
        temp['df'] -= temp['df'].min()
        po = temp.df[temp.pdir == temp.ori].values[0]
        o_deg1 = (temp.pdir-90) % 360
        o_deg2 = (temp.pdir+90) % 360
        o1 = temp.df[temp.ori == o_deg1].values[0]
        o2 = temp.df[temp.ori == o_deg2].values[0]
        oo = np.mean([o1, o2])
        osi = _osi(po,oo)
        osis.append(osi)

    # osis = abs(np.array(osis))
    # osis = abs(osis[~np.isnan(osis)])

    osi = pd.Series(osis, name='dsi')

    return osi

def osi(mdf:pd.DataFrame):
    tc = get_tc(mdf, drop_grey_screen=True, ori=False)
    
    osis = []
    for cell in tc.cell.unique():
        temp = tc[tc.cell == cell].copy()
        
        temp['df'] -= temp['df'].min()
        
        po_resp = temp.df[temp.pref == temp.ori].values[0]
        oo_resp = temp.df[temp.ortho == temp.ori].values[0]

        osi = _osi(po_resp,oo_resp)
        osis.append(osi)

    # osis = abs(np.array(osis))
    # osis = abs(osis[~np.isnan(osis)])

    osi = pd.Series(osis, name='osi')
    
    return osi

def osi_vecsum(mdf:pd.DataFrame):
    tc = get_tc(mdf, drop_grey_screen=True, ori=False)
    
    osis = []
    for cell in tc.cell.unique():
        temp = tc[tc.cell == cell].copy()
        temp['df'] -= temp['df'].min()
        dirs = temp.ori.unique()
        vals = temp['df']
        osi = _osi_vector_sum(vals, dirs)
        osis.append(osi)
    osi = pd.Series(osis, name='osi_vecsum')
    return osi

def _osi(preferred_responses, ortho_responses):
    """This is the hard-coded osi function."""
    return ((preferred_responses - ortho_responses)
            / (preferred_responses + ortho_responses))
    
def _osi_vector_sum(mean_tc, dirs):
    return get_osi_vecsum(mean_tc, dirs)
    
def _global_osi(tuning_curve):
    # TODO
    pass

def get_osi_vecsum(arr, dirs):
    th = np.deg2rad(np.mod(dirs,180))
    sinterm = np.sin(2*th).dot(arr)
    costerm = np.cos(2*th).dot(arr)
    sumterm = np.sum(arr)
    val = np.sqrt(costerm**2 + sinterm**2)/sumterm
    return val

def get_cmi(cross, iso):
    return ((cross-iso)/(cross+iso))

def get_smi(ctr, surr):
    # return ((ctr-surr)/(ctr+surr))
    return ((ctr-surr)/(surr))
    # return ctr/surr


def get_size_tc(mdf: pd.DataFrame, return_err=False):
    d = mdf.copy()
    m = d.groupby(['cell', 'sz']).mean().reset_index()
    e = d.groupby(['cell', 'sz']).mean().reset_index()
    if return_err:
        return m,e
    else:
        return m

def get_pref_size(mdf: pd.DataFrame) -> pd.Series:
    sz_df = get_size_tc(mdf)
    pref_sizes = sz_df.set_index('sz').groupby('cell')['df'].idxmax()
    pref_sizes.name = 'pref_size'
    return pref_sizes

def get_ssi(mdf: pd.DataFrame) -> pd.Series:
    resps = get_size_tc(mdf)
    # resps_at_pref = resps.loc[resps.sz == resps.pref_size, 'df'].to_numpy()
    resps_at_smallest = resps.loc[resps.sz == resps.sz.min(), 'df'].to_numpy()
    resps_at_largest = resps.loc[resps.sz == resps.sz.max(), 'df'].to_numpy()
    # ssi = get_smi(resps_at_pref, resps_at_largest)
    ssi = get_smi(resps_at_smallest, resps_at_largest)
    return pd.Series(ssi, name='ssi')
