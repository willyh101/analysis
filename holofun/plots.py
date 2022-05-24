from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from .stats import sem

def scatter2hist(x, y, bins=(10,10), ax=None, do_log=False, **kwargs):
    """Generate a 2D histogram from scatter plot data. Bins can be int or tuple."""
    if ax is None:
        ax = plt.gca()
    h, *_ = np.histogram2d(x, y, bins=bins)
    if do_log:
        h = np.log(h)
    ax.imshow(np.rot90(h), interpolation='nearest', **kwargs)
    return ax
    
def plot_mean_dff(trwise_data, cells=None, trials=None, xvals=None, ax=None, **kwargs):
    if ax is None:
       fig, ax = plt.subplots(1,1, figsize=(4,4), constrained_layout=True)
       
    if cells is None:
        cells = np.arange(trwise_data.shape[1])
    if trials is None:
        trials = np.arange(trwise_data.shape[0])
    
    mask = np.ix_(trials, cells)
    
    mm = trwise_data[mask].mean(0).mean(0)
    err = sem(trwise_data[mask].mean(0), 0)
    
    if xvals is None:
        x = np.arange(mm.size)
    else:
        x = xvals
    
    ax.plot(x, mm, **kwargs)
    ax.fill_between(x, mm+err, mm-err, alpha=0.5, **kwargs)
    
    return ax

def plot_mean_dff_by_cell(trwise_data, cells=None, trials=None, xvals=None, ax=None):
    if ax is None:
       fig, ax = plt.subplots(1,1, figsize=(4,4), constrained_layout=True)
       
    if cells is None:
        cells = np.arange(trwise_data.shape[1])
    if trials is None:
        trials = np.arange(trwise_data.shape[0])
    
    mask = np.ix_(trials, cells)
    
    mm = trwise_data[mask].mean(0)
    err = sem(trwise_data[mask], 0)
    
    if xvals is None:
        x = np.arange(mm.size)
    else:
        x = xvals
    
    for c,m,e in zip(cells, mm, err):
        ax.plot(x, m, label=c)
        ax.fill_between(x, m+e, m-e, alpha=0.5)

    return ax

# this might not be what I want exactly since you might want to adjust params based on how big
# the source image is. this is made for an 8x8
def add_scalebar(ax, px_length, um_length):
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData, 
                            px_length,
                            f'{um_length} $\mu$m',
                            'lower right',
                            pad=0.2,
                            color='white',
                            frameon=False,
                            size_vertical=8,
                            fontproperties=fontprops)
    return scalebar