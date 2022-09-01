import pandas as pd
import scipy.stats as stats

from .analysis import make_mean_df
from .traces import (baseline_subtract, cut_psths, make_trialwise,
                     min_subtract, rolling_baseline_dff, unravel, reravel)
from .vis.generic import find_vis_resp
from .vis.tuning import osi, pdir, po


def process_s2p(s2p, epoch, pre_time, total_time=None, do_zscore=False):
    # get traces
    raw_traces = s2p.cut_traces_epoch(epoch)
    traces = min_subtract(raw_traces)
    
    # calculate rolling baseline
    rwin = int(s2p.fr * 60)
    traces = rolling_baseline_dff(traces, window=rwin)
    
    if do_zscore:
        traces = stats.zscore(traces, axis=1)

    # make trialwise
    lengths = s2p.get_epoch_trial_lengths(epoch)
    trwise = make_trialwise(traces, lengths)
    trwise = baseline_subtract(trwise, pre_time)
    
    if total_time is not None:
        trwise = cut_psths(trwise, total_time)
        
    return traces, trwise

def process_oasis(s2p, epoch, pre_time, penalty=0, optimize_g=True, as_trialwise=True):
    """Run the standard suite2p pipeline and then process with oasis."""
    from .deconvolution import run_oasis
    
    _, trwise = process_s2p(s2p, epoch, pre_time)
    tr_flat = unravel(trwise)
    c,s,p = run_oasis(tr_flat, penalty=penalty, optimize_g=optimize_g)
    
    if as_trialwise:
        min_length = min(s2p.get_epoch_trial_lengths(epoch))
        c = reravel(c, min_length)
        s = reravel(s, min_length)
        p = reravel(p, min_length)
    
    return c,s,p

def ori_vis_pipeline(df, analysis_window):
    """
    Runs the visual analysis pipeline on a DataFrame. Creates and returns
    the mean DataFrame, finds visually responsive cells, finds preferred and
    ortho orientations (and preferred direction), and calculates OSI. Appends
    and returns values to the original DataFrame and mean DataFrame.

    Args:
        df (pd.DataFrame): the input DataFrame
        analysis_window (tuple): start and stop time of window to get mean response from
        col_name (str): Name of column to calculate mean from.

    Returns:
        2 dataframes, mean and original with values appended.
    """
    
    mdf = make_mean_df(df, analysis_window, 'ori')

    cells, pvals = find_vis_resp(mdf)
    prefs, orthos = po(mdf)
    pdirs = pdir(mdf)
    # odirs = odir(mdf)

    mdf.loc[:, 'vis_resp'] = False
    mdf.loc[mdf.cell.isin(cells), 'vis_resp'] = True

    mdf = mdf.join(pd.Series(pvals, name='pval'), on='cell')

    # since these functions returned pd.Series not going to bother with df_add_trialwise/cellwise
    mdf = mdf.join(prefs, on='cell')
    mdf = mdf.join(orthos, on='cell')
    mdf = mdf.join(pdirs, on='cell')

    osis = osi(mdf)
    mdf = mdf.join(osis, on='cell')

    df = df.join(prefs, on='cell')
    df = df.join(orthos, on='cell')
    df = df.join(pdirs, on='cell')
    df = df.join(osis, on='cell')
    
    df.loc[:, 'vis_resp'] = False
    df.loc[df.cell.isin(cells), 'vis_resp'] = True
    df = df.join(pd.Series(pvals, name='pval'), on='cell')
    
    return df, mdf
