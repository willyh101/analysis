import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import axes
import logging

from .stats import ci, sem, fit_fast
from .vis.retinotopy import Retinotopy


MPL_RC_DEFAULT = {
    'figure.figsize': (4,3),
    'figure.constrained_layout.use': True
}


def scatter2hist(x, y, bins=(10,10), ax=None, do_log=False, **kwargs):
    """Generate a 2D histogram from scatter plot data. Bins can be int or tuple."""
    if ax is None:
        ax = plt.gca()
    h, *_ = np.histogram2d(x, y, bins=bins)
    if do_log:
        h = np.log(h)
    ax.imshow(np.rot90(h), interpolation='nearest', **kwargs)
    return ax

def line_fill(x, y, yerr, ax=None, label=None, line_color=None, fill_color=None, alpha=0.5, **kwargs):
    """Create a line plot with a filled error region."""
    color = kwargs.pop('color', None)
    if color is not None:
        line_color = color
        fill_color = color
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, label=label, color=line_color, **kwargs)
    ax.fill_between(x, y-yerr, y+yerr, alpha=alpha, color=fill_color, **kwargs)
    return ax
    
def plot_mean_dff(trwise_data: np.ndarray, cells=None, trials=None, xvals=None, fr=None, ax=None, 
                  falpha=0.5, label=None, **kwargs):
    if ax is None:
    #    fig, ax = plt.subplots(1,1, figsize=(4,3), constrained_layout=True)
        fig = plt.figure(figsize=(3,2))
        ax = fig.subplots(1,1)
       
    if cells is None:
        cells = np.arange(trwise_data.shape[1])
    if trials is None:
        trials = np.arange(trwise_data.shape[0])
    
    mask = np.ix_(trials, cells)
    
    mm = trwise_data[mask].mean(0).mean(0)
    err = sem(trwise_data[mask].mean(0), 0)
    
    if xvals is None and fr is None:
        x = np.arange(mm.size)
    elif xvals is None and fr is not None:
        x = np.arange(mm.size) / fr
    else:
        x = xvals
    
    ax.plot(x, mm, label=label, **kwargs)
    ax.fill_between(x, mm+err, mm-err, edgecolor=None, alpha=falpha, **kwargs)
    ax.set_ylabel('∆F/F')
    ax.set_xlabel('Time (s)')
    
    return ax

def plot_mean_dff_by_cell(trwise_data: np.ndarray, cells=None, trials=None, xvals=None, fr=None, ax=None, **kwargs):
    if ax is None:
       fig, ax = plt.subplots(1,1, figsize=(3,3), constrained_layout=True)
       
    if cells is None:
        cells = np.arange(trwise_data.shape[1])
    if trials is None:
        trials = np.arange(trwise_data.shape[0])
    
    mask = np.ix_(trials, cells)
    
    mm = trwise_data[mask].mean(0)
    err = sem(trwise_data[mask], 0)
    
    if xvals is None and fr is None:
        x = np.arange(mm.shape[-1])
    elif xvals is None and fr is not None:
        x = np.arange(mm.shape[-1]) / fr
    else:
        x = xvals
    
    for c,m,e in zip(cells, mm, err):
        ax.plot(x, m, label=c, **kwargs)
        ax.fill_between(x, m+e, m-e, alpha=0.5, **kwargs)

    return ax

def plot_mean_dff_of_cell(trwise_data: np.ndarray, cell: int, trials=None, 
                          xvals=None, fr=None, ax=None, label=None, **kwargs):
    if ax is None:
       fig, ax = plt.subplots(1,1, figsize=(3,3), constrained_layout=True)
       
    if trials is None:
        trials = np.arange(trwise_data.shape[0])
    
    mask = np.ix_(trials, [cell])
    
    mm = trwise_data[mask].mean(0).squeeze()
    err = sem(trwise_data[mask], 0).squeeze()
    
    if xvals is None and fr is None:
        x = np.arange(mm.size)
    elif xvals is None and fr is not None:
        x = np.arange(mm.size) / fr
    else:
        x = xvals
    
    ax.plot(x, mm, label=label, **kwargs)
    ax.fill_between(x, mm+err, mm-err, alpha=0.5, **kwargs)

    return ax

def plot_tc(d: pd.DataFrame, cell: int, drop_grey_screen=True, 
            ax: axes.Axes = None, **kwargs):
    """Give a mean DataFrame and a cell number, plot the tuning curve."""
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(3,3), constrained_layout=True)
        
    if drop_grey_screen:
        vals = d.loc[d.ori >= 0].copy()
    else:
        vals = d.copy()
        
    m = vals[vals.cell == cell].groupby('ori').mean()['df']
    e = vals[vals.cell == cell].groupby('ori').sem()['df']
    xs = vals[vals.cell == cell].groupby('ori').mean().index.astype(int)
    
    kwargs.setdefault('linewidth', 2)
    
    ax.errorbar(xs, m, e, **kwargs)
    ax.set_ylabel('∆F/F')
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, rotation=-45)
    
    return ax

def plot_cell_ret(ret_data: np.ndarray, ret_fit: Retinotopy):
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

def get_all_axes():
    return plt.gcf().get_axes()
    
def update_all(axes: np.ndarray = None, **kwargs):
    all_axes = get_all_axes()
    for ax in all_axes:
        for k,v in kwargs.items():
            fxn = eval(f'ax.{k}')
            fxn(v)

def remove_ticks():
    all_axes = get_all_axes()
    for ax in all_axes:
        ax.set_xticks([])
        ax.set_yticks([])

def restore_spines():
    all_axes = get_all_axes()
    for ax in all_axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

def remove_spines():
    all_axes = get_all_axes()
    for ax in all_axes:
        for side in ['top', 'right', 'left', 'bottom']:
            ax.spines[side].set_visible(False)
            
def scatter_eq_axis(x: np.ndarray, y:np.ndarray, xy_max=None, xy_min=0, fit=False, fit_color='r', 
                    nticks=3, kde=False, add_mean=False, mean_est=ci, ax=None, **kwargs) -> axes.Axes:
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
        _, xy_max = calculate_xy_limits(x, y)
    
    if xy_min is None:
        xy_min, _ = calculate_xy_limits(x, y)
        
    if kde:
        x,y,cdata = estimate_kde(x,y)
        kwargs.pop('color',None)
        kwargs['c'] = cdata
    
    ax.scatter(x, y, **kwargs)
    ax.plot([xy_min,xy_max], [xy_min,xy_max], c='k', ls='--')
        
    if add_mean:
        ax.errorbar(np.nanmean(x), np.nanmean(y), xerr=mean_est(x), yerr=mean_est(y), 
                    fmt='-o', color=kwargs['color'])
    
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    if nticks:
        ax.locator_params(axis='both', nbins=nticks)
    ax.set_aspect('equal', 'box')
    
    if fit:
        try:
            xrng, yfit = fit_fast(x, y)
            ax.plot(xrng, yfit, c=fit_color)
        except:
            logging.warning('Failed to fit line to data.')
        
    return ax

def calculate_xy_limits(x, y, modifier=1.1):
    data_max = np.nanmax([x,y])
    data_min = np.nanmin([x,y])
    data_range = data_max - data_min
    adj_min = data_min - (data_range * (modifier-1))
    adj_max = data_max + (data_range * (modifier-1))
    return adj_min, adj_max

def estimate_kde(x: np.ndarray, y: np.ndarray):
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x,y,z = xy[0,idx], xy[1,idx], z[idx]
    return x,y,z

def df_scatter_eq(x: str, y: str, data: pd.DataFrame, kde=False, log=False, **kwargs) -> axes.Axes:
    """
    Creates a scatter plot from a pandas DataFrame with equal axis sizes.

    Parameters:
        x (str): The column name in the DataFrame for the x-axis data.
        y (str): The column name in the DataFrame for the y-axis data.
        data (pd.DataFrame): The DataFrame containing the data to plot.
        kde (bool, optional): Whether to estimate and color the data points by their kernel density. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the scatter plot function.

    Returns:
        axes.Axes: The matplotlib Axes object containing the scatter plot.
    """
    xdata = data.loc[:,x]
    ydata = data.loc[:,y]
    
    if log:
        xdata = np.log(xdata)
        ydata = np.log(ydata)
    
    if kde:
        xdata,ydata,cdata = estimate_kde(xdata,ydata)
        kwargs.pop('color', None)
        kwargs['c'] = cdata
        
    ax = scatter_eq_axis(xdata, ydata, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    return ax

def histfill(vals, bw=100, ax=None, fill_alpha=0.4, label=None, color=None, **kwargs):
    """
    Plots a histogram of the given values with filled bars.

    Parameters:
        vals (array-like): The values to be plotted.
        bw (int, optional): The number of bins in the histogram. Default is 100.
        ax (matplotlib.axes.Axes, optional): The axes on which the histogram is plotted. If not provided, the current axes are used.
        fill_alpha (float, optional): The transparency of the filled bars. Default is 0.4.
        label (str, optional): The label for the plot.
        color (str, optional): The color of the plot.
        **kwargs: Additional keyword arguments to be passed to the plot.

    Returns:
        matplotlib.axes.Axes: The axes object with the plotted histogram.
    """
    if ax is None:
        ax = plt.gca()
    ys, xs = np.histogram(vals, bw)
    ax.fill_between(xs[1:], ys, alpha=fill_alpha, color=color, **kwargs)
    ax.plot(xs[1:], ys, label=label, color=color)
    return ax

def jitter_xy(y, cat_idx, jitter=0.1):
    """Jitter x values by a small amount for plotting categorical data."""
    rng = np.random.default_rng()
    y = np.asarray(y)
    n = len(y)
    x = cat_idx + rng.normal(size=n) * jitter
    return x,y

def paired_plot(a=None, b=None, data=None, ax=None, show_pval=True, use_ttest=False, **kwargs):
    """Paired comparison plot with error bars."""
    if ax is None:
        ax = plt.gca()
    if isinstance(a, pd.DataFrame) and b is None:
        data = a
    if data is None:
        data = pd.DataFrame(np.array([a,b]).T, columns=['A','B'])
        
    kwargs.setdefault('c', 'tab:blue')
    kwargs.setdefault('alpha', 0.5)
    
    ax.plot(data.T, **kwargs)
    ax.errorbar(y=[*data.mean()], x=[0,1], yerr=[*data.agg(ci)], c='k', lw=2)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(data.min().min(), data.max().max()*1.2)
    ax.set_xticks([0,1])
    
    if show_pval:
        if use_ttest:
            result = stats.ttest_rel(a=data.iloc[:,0], b=data.iloc[:,1])
        else:
            result = stats.wilcoxon(x=data.iloc[:,0], y=data.iloc[:,1])
        ax.text(x=0.5, y=data.max().max()*1.1, s=f'p={result.pvalue:.5f}', ha='center')
        
    return ax

def plot_means_eq(x: pd.Series, y: pd.Series, err_func='ci', ax=None, union=True, **kwargs):
    """Plot means with error bars."""
    if ax is None:
        ax = plt.gca()
    xy_min, xy_max = calculate_xy_limits(x, y)
    if err_func == 'ci':
        err_func = ci
    x_agg = x.agg(['mean', err_func])
    y_agg = y.agg(['mean', err_func])
    if union:
        ax.plot([xy_min,xy_max], [xy_min,xy_max], c='k', ls='--')
    ax.errorbar(x_agg['mean'], y_agg['mean'], xerr=x_agg.iloc[1], yerr=y_agg.iloc[1], 
                fmt='-', **kwargs)
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_aspect('equal', 'box')
    return ax

def plot_means_eq_df(data: pd.DataFrame, x: str, y: str, **kwargs):
    """Plot means with error bars from a DataFrame."""
    xvals = data.loc[:,x]
    yvals = data.loc[:,y]
    ax = plot_means_eq(xvals, yvals, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax

def simplecat(vals:list, cats=None, alpha=0.5, c='cornflowerblue', ax=None, jit=0.1, 
              skws=None, mkws=None, add_ct=None, ct_rng=0.25):
    if ax is None:
        ax = plt.gca()
    if cats is None:
        cats = np.arange(len(vals))
    if skws is None:
        skws = {}
    if mkws is None:
        mkws = {}
    mkws.setdefault('color', 'k')
    mkws.setdefault('linestyle', 'none')
    mkws.setdefault('marker', 'o')
    mkws.setdefault('markersize', 5)
    mkws.setdefault('capsize', 5)
    skws.setdefault('edgecolor', 'none')
    xs = []
    ys = []
    for i,y in enumerate(vals):
        x_, y_ = jitter_xy(y, i, jitter=jit)
        # xs.append(x_)
        # ys.append(y_)
        ax.scatter(x_, y_, color=c, alpha=alpha, **skws)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(cats)
    ax.set_xlim(-0.5, len(vals)-0.5)

    if add_ct:
        for i, v in enumerate(vals):
            xmin = i-ct_rng
            xmax = i+ct_rng
            func = eval(f'np.{add_ct}')
            ax.hlines(func(v), xmin=xmin, xmax=xmax, color='k', ls='--')    

    return ax

def simplecat_df(grp, cats, **kwargs):
    gs=[]
    for _,g in grp:
        gs.append(g.values)
    ax = simplecat(gs, cats, **kwargs)
    return ax

def catscatter(colors, alpha=0.5, ax=None, skws=None, df=None, add_ct=False, ct_rng=0.25, s=10, **data):
    if ax is None:
        ax = plt.gca()
    if isinstance(df, pd.DataFrame):
        data = df.to_dict(orient='series')
    if skws is None:
        skws = {}
    if isinstance(colors, str):
        colors = [colors]
    if len(colors) == 1:
        colors *= len(data)
        
    skws.setdefault('edgecolor', 'none')
    skws.setdefault('s', s)
    
    xs = []
    ys = []
    cs = []
    for i, (k,v) in enumerate(data.items()):
        if isinstance(v, pd.Series):
            data[k] = v.values
        x_,y_ = jitter_xy(v, i)
        xs.append(x_)
        ys.append(y_)
        # cs.extend([colors[i]]*len(v))
        # cs = colors[i]
        
        ax.scatter(x_, y_, color=colors[i], alpha=alpha, **skws)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.keys())
    ax.set_xlim(-0.5, len(data)-0.5)

    if add_ct:
        for i, (k,v) in enumerate(data.items()):
            xmin = i-ct_rng
            xmax = i+ct_rng
            func = eval(f'np.{add_ct}')
            ax.hlines(func(ys[i]), xmin=xmin, xmax=xmax, color='k', ls='--')    
    
    return ax
    
def catplot(colors, alpha=0.5, ax=None, skws=None, mkws=None, 
            err_func=ci, colors_vary=False, **data):
    if isinstance(err_func, str):
        err_func = eval(err_func)
    if ax is None:
        ax = plt.gca()
    if skws is None:
        skws = {}
    if mkws is None:
        mkws = {}
    if isinstance(colors, str):
        colors = [colors]
    if len(colors) == 1:
        colors *= len(data)
    mkws.setdefault('color', 'k')
    mkws.setdefault('linestyle', 'none')
    mkws.setdefault('marker', 'o')
    mkws.setdefault('markersize', 5)
    mkws.setdefault('capsize', 5)
    skws.setdefault('edgecolor', 'none')

    xs = []
    ys = []
    cs = []
    for i, (k,v) in enumerate(data.items()):
        if isinstance(v, pd.Series):
            data[k] = v.values
        x_,y_ = jitter_xy(v, i)
        xs.append(x_)
        ys.append(y_)
        cs.extend([colors[i]]*len(v))
        
    ax.scatter(xs, ys, color=cs, alpha=alpha, **skws)

    if not colors_vary:
        ax.errorbar(
            x=[np.nanmean(x) for x in xs], 
            y=[np.nanmean(y) for y in ys], 
            yerr=[err_func(y) for y in ys],
            **mkws
        )
    else:
        mkws.pop('color')
        for i,c in enumerate(colors):
            ax.errorbar(x=np.nanmean(xs[i]), y=np.nanmean(ys[i]), yerr=err_func(ys[i]), color=c, **mkws)
    
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.keys())
    ax.set_xlim(-0.5, len(data)-0.5)

    return ax