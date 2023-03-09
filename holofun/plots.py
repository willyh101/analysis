import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
    
def plot_mean_dff(trwise_data, cells=None, trials=None, xvals=None, ax=None, falpha=0.5, **kwargs):
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
    ax.fill_between(x, mm+err, mm-err, edgecolor=None, alpha=falpha, **kwargs)
    
    return ax

def plot_mean_dff_by_cell(trwise_data, cells=None, trials=None, xvals=None, ax=None, **kwargs):
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
        x = np.arange(mm.shape[-1])
    else:
        x = xvals
    
    for c,m,e in zip(cells, mm, err):
        ax.plot(x, m, label=c, **kwargs)
        ax.fill_between(x, m+e, m-e, alpha=0.5, **kwargs)

    return ax

def plot_mean_dff_of_cell(trwise_data, cell, trials=None, xvals=None, ax=None, **kwargs):
    if ax is None:
       fig, ax = plt.subplots(1,1, figsize=(4,4), constrained_layout=True)
       
    if trials is None:
        trials = np.arange(trwise_data.shape[0])
    
    mask = np.ix_(trials, [cell])
    
    mm = trwise_data[mask].mean(0).squeeze()
    err = sem(trwise_data[mask], 0).squeeze()
    
    if xvals is None:
        x = np.arange(mm.size)
    else:
        x = xvals
    
    ax.plot(x, mm, label=cell, **kwargs)
    ax.fill_between(x, mm+err, mm-err, alpha=0.5, **kwargs)

    return ax

def plot_tc(d, cell, drop_grey_screen=True, ax=None, **kwargs):
    """Give a mean DataFrame and a cell number, plot the tuning curve."""
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(3,3), constrained_layout=True)
        
    if drop_grey_screen:
        vals = d.loc[d.ori >= 0].copy()
        
    m = vals[vals.cell == cell].groupby('ori').mean()['df']
    e = vals[vals.cell == cell].groupby('ori').sem()['df']
    xs = vals.ori.unique()
    
    kwargs.setdefault('linewidth', 2)
    
    ax.errorbar(xs, m, e, **kwargs)
    ax.set_ylabel('$\Delta$F/F')
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, rotation=-45)
    
    return ax

def plot_cell_ret(ret_data, ret_fit):
    fig, ax = plt.subplots(1,3, figsize=(4,4))

    x,y,_ = ret_fit.calculate_grid()
    ext = (x.min(), x.max(), y.min(), y.max())
    ax[0].imshow(ret_data, extent=ext)
    ax[0].set_title('Data')
    ret_fit.plot(ax=ax[1])
    ax[1].set_title('Fit')
    ax[2].set_title('Model')
    ret_fit.plot(expand_by=100, ax=ax[2])

    for a in ax:
        a.axis('off')
        
    # fig.subplots_adjust(wspace=.01, hspace=.01)
    plt.show()
    
def update_all(axes, **kwargs):
    for ax in axes.ravel():
        for k,v in kwargs.items():
            fxn = eval(f'ax.{k}')
            fxn(v)
            
def scatter_eq_axis(x, y, xy_max=None, xy_min=0, fit=False, fit_color='r', nticks=3, kde=False,
                    ax=None, **kwargs):
    """
    Plot a scatter plot relating X and Y with equal axis sizes and optionally a reference line and
    optionally a fit line.

    Args:
        x (array-like): x-data
        y (array-like): y-data
        xy_max (int/float): max value for x and y limits
        xy_min (int, optional): min value for x and y limits. Defaults to 0.
        fit (bool, optional): Whether to determine the linear fit. Defaults to False.
        fit_color (str, optional): Color of fitline if enabled. Defaults to 'r'.
        nticks (int, optional): Number of ticks to plot on each axis. Defaults to 4.
        ax (matplotlib.axes, optional): Axes to plot on. Defaults to None.

    Returns:
        modified plotting axes
    """
    if ax is None:
        ax = plt.gca()
        
    kwargs.setdefault('s', 6)
    
    if xy_max is None:
        xy_max = np.nanmax([x,y]) * 1.1
        
    if kde:
        x,y,cdata = estimate_kde(x,y)
        kwargs.pop('color',None)
        kwargs['c'] = cdata
    
    ax.scatter(x, y, **kwargs)
    ax.plot([xy_min,xy_max], [xy_min,xy_max], c='k', ls='--')
    
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    if nticks:
        ax.locator_params(axis='both', nbins=nticks)
    ax.set_aspect('equal', 'box')
    
    if fit:
        m,b = np.polyfit(x, y, 1)
        xrng = np.arange(0,xy_max,0.01)
        yfit = (m*xrng)+b
        ax.plot(xrng, yfit, c=fit_color)
        
    return ax

def estimate_kde(x, y):
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x,y,z = xy[0,idx], xy[1,idx], z[idx]
    return x,y,z

def df_scatter_eq(x, y, data, kde=False, **kwargs):
    xdata = data.loc[:,x]
    ydata = data.loc[:,y]
    
    if kde:
        xdata,ydata,cdata = estimate_kde(xdata,ydata)
        kwargs.pop('color')
        kwargs['c'] = cdata
        
    ax = scatter_eq_axis(xdata, ydata, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    return ax