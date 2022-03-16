from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from .stats import sem

def scatter2hist(x, y, bins=(10,10), ax=None, do_log=False):
    """Generate a 2D histogram from scatter plot data. Bins can be int or tuple."""
    if ax is None:
        ax = plt.gca()
    h, *_ = np.histogram2d(x, y, bins=bins)
    if do_log:
        h = np.log(h)
    ax.imshow(np.rot90(h), interpolation='nearest')
    return ax

def traces_fill_plot(mm, err, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(mm)
    ax.fill_between(np.arange(mm.size), mm+err, mm-err, alpha=0.5)
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

def plot_mean_dff(trwise_data, ax=None, title=None):
    if ax is None:
       fig, ax = plt.subplots(1,1, figsize=(4,4), constrained_layout=True)
    
    mm = trwise_data.mean(0).mean(0)
    err = sem(trwise_data.mean(0), 0)
    x = np.arange(mm.size)
    
    ax.plot(x,mm)
    ax.fill_between(x, mm+err, mm-err, alpha=0.5)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('$\Delta$F/F')
    
    return ax