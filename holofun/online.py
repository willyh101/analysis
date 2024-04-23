import pandas as pd
import numpy as np

def traces_from_csv(online_path, nplanes):
    """Extract raw F traces from online data. No post-processing."""
    online_data = pd.read_csv(online_path)
    online_data = online_data.set_index('frameNumber')
    new_idx = list(range(nplanes, online_data.index.max(), nplanes))
    online_data_reindex = online_data.reindex(new_idx, index='frameNumber', method='nearest')
    online_arr=online_data_reindex.values[:,1:]
    online_arr=online_arr.T
    return online_arr

def make_online_trialwise(traces, trial_lengths):
    """Returns trial x cell x time for online data."""
    traces = np.split(traces, np.cumsum(trial_lengths[:-1]), axis=1)
    shortest = min([s.shape[1] for s in traces])
    return np.array([a[:, :shortest] for a in traces])