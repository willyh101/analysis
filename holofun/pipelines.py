from .traces import min_subtract, rolling_baseline_dff, make_trialwise
import scipy.stats as stats
from .analysis import make_mean_df
import pandas as pd
from .vis import find_vis_resp, po, pdir, osi

def baseline_and_zscore(raw_traces):
    traces = min_subtract(raw_traces)
    traces_b = rolling_baseline_dff(traces)
    traces_z = stats.zscore(traces_b, axis=1)
    return traces_z

def make_trialwise_traces(s2p_data, epoch):
    raw_traces = s2p_data.cut_traces_epoch(epoch)
    traces = baseline_and_zscore(raw_traces)
    lengths = s2p_data.get_epoch_trial_lengths(epoch)
    trialwise_data = make_trialwise(traces, lengths)
    return trialwise_data
    

# def process_s2p_epoch(s2p_obj, epoch):
#     # first, get traces
#     traces = s2p_obj.cut_traces_epoch(epoch)
#     traces_scored = baseline_and_zscore(traces)
    
# class s2pEpoch:
#     def __init__(self, s2p, epoch):
#         self.s2p = s2p
#         self.epoch = epoch
        
#         self.raw_traces = s2p.cut_traces_epoch(epoch)
        
#     def baseline_and_zscore(self):
#         traces = min_subtract(self.raw_traces)
#         traces_b = rolling_baseline_dff(traces)
#         traces_z = stats.zscore(traces_b, axis=1)
#         return traces_z
    
#     def make_trialwise(self):
        

# def make_trialwise_aligned_df(traces, trial_lengths, stim_times, align_to, stims):
#     trwise = make_trialwise(traces, trial_lengths)
#     aligned = make_psths(trwise, stim_times, align_to)
#     df = make_dataframe(aligned, fr, stims, )

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