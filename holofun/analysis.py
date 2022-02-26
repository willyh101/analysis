import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.neighbors import KDTree

def make_mean_df(df, win, col):
    """
    Takes the mean by a condition in the data frame betweeen 2 timepoints and
    returns a mean dataframe reduced over the column condition.

    Inputs:
        df: the dataframe
        win (tuple): start and end time in whatever 'time' is in the dataframe
        col (str): column name that you are meaning over

    Returns:
        mean dataframe
    """

    # implemented trialwise subtraction
    assert len(win) == 4, 'Must give 4 numbers for window.'
    base = df[(df.time > win[0]) & (df.time < win[1])].groupby(['cell', col, 'trial']).mean().reset_index()
    resp = df[(df.time > win[2]) & (df.time < win[3])].groupby(['cell', col, 'trial']).mean().reset_index()
    resp['df'] = resp['df'] - base['df']
    return resp

def coords2cells(coords, meds, threshold=15):
    """
    Single plane method with KDTree. Avoid optotune <-> holography mismatch weirdness.
    Threshold is in pixels.
    """
    holo_zs = np.unique(coords[:,2])
    opto_zs = np.unique(meds[:,2])

    matches = []
    distances = []
    ismatched = []

    for hz, oz in zip(holo_zs, opto_zs):
        this_plane_meds = meds[meds[:,2] == oz]
        this_plane_targs = coords[coords[:,2] == hz]
        
        cells = KDTree(this_plane_meds[:,:2])
        dists, match_idx = cells.query(this_plane_targs[:,:2])
        dists = dists.squeeze()
        match_idx = match_idx.squeeze()[dists < threshold]
        
        matches.append(this_plane_meds[match_idx,:])
        distances.append(dists[dists < threshold])
        ismatched.append(dists < threshold)
    
    
    locs = np.vstack(matches)
    distances = np.concatenate(distances)
    ismatched = np.concatenate(ismatched)
    
    return locs, distances, ismatched

def match_cells(meds, targs, threshold=15):
    """
    Shortcut to quickly identify the s2p cells that are matched to targets.

    Args:
        meds (np.array): all suite2p sources
        targs (np.array): target cells to match`
        threshold (int, optional): cut-off distance for matching. Defaults to 10.

    Returns the indices of s2p matched cells.
    """
    locs, *_ = coords2cells(targs, meds, threshold)
    matches = np.concatenate([np.where(np.all(meds == locs[m,:], axis=1))[0] for m in range(locs.shape[0])])
    return matches
    

def find_responsive_cells(df, col, cond, win, one_way=False, test_fxn='ttest', pval=0.05):
    
    test_funs = {
        'ttest': _ttest_rel_df,
        'wilcoxon': _wilcoxon_df,
        'increase': _increase_test_df,
        'ranksum': _ranksum_df,
    }
    
    test = test_funs[test_fxn]
    
    base = df[(df.time > win[0]) & (df.time < win[1])].groupby(['cell', col, 'trial']).mean().reset_index()
    resp = df[(df.time > win[2]) & (df.time < win[3])].groupby(['cell', col, 'trial']).mean().reset_index()['df']

    base = base.rename(columns={'df':'baseline'})
    resp = resp.rename('resp')
    
    resp_df = pd.concat((base,resp), axis=1)

    ps = resp_df[resp_df[col] == cond].groupby('cell').apply(lambda x: test(x)).values
    
    if one_way:
        ps /= 2
    
    cells = np.where(ps < pval)[0]
    
    return ps, cells

def id_cells_in_df(df, cells, new_col_name):
    """given a list of cells, assign them as 'True' in a new column on a df"""
    if new_col_name in df:
        df = df.drop([new_col_name], axis=1)
    df.loc[:,new_col_name] = False
    df.loc[df['cell'].isin(cells), new_col_name] = True
    return df

def _ttest_rel_df(grp):
    return stats.ttest_rel(grp['baseline'], grp['resp'])[1]

def _wilcoxon_df(grp):
    return stats.wilcoxon(grp['baseline'], grp['resp'])[1]

def _ranksum_df(grp):
    return stats.ranksums(grp['baseline'], grp['resp'])[1]

def _increase_test_df(grp):
    b = grp['baseline'].mean()
    r = grp['resp'].mean()
    if r > 3*b:
        p = 0.01
    else:
        p = 1
    return p