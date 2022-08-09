import numpy as np
import scipy.signal as sig
from oasis.functions import deconvolve


def run_oasis(traces, penalty=0, optimize_g=True):
    c_traces = np.zeros_like(traces)
    s_traces = np.zeros_like(traces)
    p_traces = np.zeros_like(traces).astype(int)
    if traces.ndim == 1:
        c_traces, s_traces, *_ = deconvolve(traces, penalty=penalty, optimize_g=optimize_g)
    else:
        for i in range(traces.shape[0]):  # cellwise
            c, s, b, g, lam = deconvolve(traces[i,:], penalty=penalty, optimize_g=optimize_g)
            c_traces[i,:] = c
            s_traces[i,:] = s
            peak_times = sig.find_peaks(s, threshold=0.1)[0]
            p_traces[i, peak_times] = 1
    return c_traces, s_traces, p_traces
