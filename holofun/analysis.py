import numpy as np
import pandas as pd
import scipy.optimize as sop
import scipy.stats as stats
from sklearn.neighbors import KDTree

from holofun.constants import PX_PER_UM, UM_PER_PIX
from holofun.traces import df_add_cellwise


def make_mean_df(df, win, col):
    """
    Takes the mean by a condition in the data frame betweeen 2 timepoints and
    returns a mean dataframe reduced over the column condition.

    Inputs:
        df: the dataframe
        win (tuple): start and end time in whatever 'time' is in the dataframe
        col (str): column name that you are meaning over, can be a list

    Returns:
        mean dataframe
    """
    # ensure col is a list for unpacking
    if not isinstance(col, list):
        col = [col]

    # implemented trialwise subtraction
    df = df.copy()
    assert len(win) == 4, 'Must give 4 numbers for window.'
    base = df[df.time.between(win[0], win[1])].groupby(['cell', *col, 'trial']).mean().reset_index()
    resp = df[df.time.between(win[2], win[3])].groupby(['cell', *col, 'trial']).mean().reset_index()
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
        ismatched.append(np.atleast_1d(dists < threshold))
    
    
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

def match_cells_hungarian(meds, targs, threshold=15):
    # quick hack for multiplane
    meds = meds.copy()
    targs = targs.copy()
    meds[:,2] *= 30
    targs[:,2] *= 30
    ds = []
    for t in targs:
        dist = np.linalg.norm(meds-t, axis=1)
        ds.append(dist)
    ds = np.array(ds)

    hungm = sop.linear_sum_assignment(ds)
    hungd = ds[hungm]
    below_thresh = hungd < threshold
    _,hungi = hungm
    return hungi[below_thresh], below_thresh.nonzero()[0]

def hungarian_matching(subset, superset):
    # compute distance matrix for each match to make cost function
    ds = []
    for t in subset:
        dist = np.linalg.norm(superset-t, axis=1)
        ds.append(dist)
    ds = np.array(ds)
    return sop.linear_sum_assignment(ds), ds

def find_responsive_cells(df, col, cond, win, one_way=False, test_fxn='ttest', alpha=0.05):
    
    test_funs = {
        'ttest': _ttest_rel_df,
        'wilcoxon': _wilcoxon_df,
        'increase': _increase_test_df,
        'ranksum': _ranksum_df,
    }
    
    test = test_funs[test_fxn]
    
    base = df[(df.time > win[0]) & (df.time < win[1])].groupby(['cell', col, 'trial']).mean(numeric_only=True).reset_index()
    resp = df[(df.time > win[2]) & (df.time < win[3])].groupby(['cell', col, 'trial']).mean(numeric_only=True).reset_index()['df']

    base = base.rename(columns={'df':'baseline'})
    resp = resp.rename('resp')
    
    resp_df = pd.concat((base,resp), axis=1)

    ps = resp_df[resp_df[col] == cond].groupby('cell').apply(lambda x: test(x)).values
    
    if one_way:
        ps /= 2
    
    resp_df = df_add_cellwise(resp_df, ps, 'pval')
    cells = resp_df[resp_df['pval'] < alpha].cell.unique()
    # cells = np.where(ps < pval)[0]
    
    return ps, cells

def id_cells_in_df(df: pd.DataFrame, cells: np.ndarray, new_col_name: str):
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

def px2mu(d):
    return d*UM_PER_PIX

def mu2px(d):
    return d*PX_PER_UM

def get_targeted_cells(target_locs, cell_locs, threshold, verbose=True):
    *_, is_match_target = coords2cells(target_locs, cell_locs, threshold)
    target_matches = match_cells(cell_locs, target_locs, threshold)
    targeted_cells = np.full_like(is_match_target, np.nan, dtype=float)
    targeted_cells[is_match_target] = target_matches
    
    if verbose:
        num_matches = np.count_nonzero(~np.isnan(targeted_cells))
        num_cells = targeted_cells.size
        target_matched_percent = num_matches/num_cells
        print(f'Matched {target_matched_percent:.2%}% of cells ({num_matches}/{num_cells})')
    return targeted_cells

def get_holo_rois(targeted_cells, daq_roi_list, verbose=True):
    holos = [targeted_cells[r-1] for r in daq_roi_list] # -1 is for matlab fix
    holos = [np.array([h]) if isinstance(h, float) else h for h in holos]
    hfunc = lambda x: np.count_nonzero(~np.isnan(x))/len(x)
    holo_match_percent = np.array(list(map(hfunc, holos)))
    
    if verbose:
        mi = holo_match_percent.min()
        mx = holo_match_percent.max()
        mn = holo_match_percent.mean()
        print(f'Holos on average matched {mn:.2%} of rois.')
        print(f'Min = {mi:.2%}')
        print(f'Max = {mx:.2%}')
    return holos, holo_match_percent

def off_targets_by_distance(s2p_locs, targ_locs, threshold=10):
    """This only works if the planes are adjusted to real distance."""
    off_target_risk = []
    for loc in targ_locs:
        plane = loc[2]
        ds = np.linalg.norm(s2p_locs-loc, axis=1) < threshold
        ps = s2p_locs[:,2] == plane
        ot = np.where(ds & ps)[0]
        off_target_risk.append(ot)
    return np.concatenate(off_target_risk)

def subsample_trials(df: pd.DataFrame, frac=0.5, replace=False):
    ntrials = df.trial.nunique()
    nchoose = int(ntrials*frac)
    sample1 = np.random.choice(df.trial.unique(), nchoose, replace=replace)
    sample2 = df.loc[~df['trial'].isin(sample1), 'trial'].unique()
    return sample1, sample2

def subsample_df(df: pd.DataFrame, frac=0.5, replace=False):
    ss = subsample_trials(df, frac, replace)
    s1 = df[(df.trial.isin(ss[0]))]
    s2 = df[(df.trial.isin(ss[1]))]
    return s1, s2

def split_and_corr_cells(df: pd.DataFrame, col: str, frac=0.5, replace=False):
    """Split trials (same for all cells) and correlate mean responses."""
    s1, s2 = subsample_df(df, frac, replace)
    s1_means = s1.groupby(['cell', col]).mean()['df']
    s2_means = s2.groupby(['cell', col]).mean()['df']
    mrg = s1_means.to_frame().join(s2_means, rsuffix='2').reset_index()
    corr_vals = mrg.groupby(['cell'])[['df','df2']].corr().iloc[0::2,-1].values
    return corr_vals

def single_cell_stim_distance(df: pd.DataFrame, cells: list[int], cols: list[str] = ['x','y'], 
                              cell_key: str = 'cell', out_str: str = 'dstim') -> pd.DataFrame:
    """Calculate the distance to a single cell of interest. Adds columns to df as dstim{cell}."""
    cols_select = [cell_key] + cols
    stim_d_xy = df.loc[df[cell_key].isin(cells), cols_select].groupby(cell_key).first()
    for i in cells:
        df[f'{out_str}{i}'] = np.linalg.norm(df[cols] - stim_d_xy.loc[i], axis=1)
    return df

def fit_xy_line(x: np.ndarray, y: np.ndarray):
    """Fit a line to x,y data. Returns general x and y."""
    m, b = np.polyfit(x, y, 1)
    xr = np.arange(x.min()*0.9, x.max()*1.1, 0.1)
    yfit = m*xr + b
    return xr, yfit