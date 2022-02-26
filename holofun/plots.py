from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def make_ticks_good(ax, psths, pre_time):
    xticks = np.arange(-pre_time/6)
    
def stripplot(x, y, data, hue=None, dodge=None, xlabel=None, ylabel=None,
                title=None, legend=None, strip_kws=None, point_kws=None, ax=None):
    if ax is None:
        ax = plt.gca()

    common_opts = {
        'x': x,
        'y': y,
        'data': data,
        'hue': hue,
        'dodge': dodge
    }

    if common_opts['hue'] and common_opts['dodge'] is None:
        common_opts['dodge'] = 0.4

    if strip_kws is None:
        strip_kws = {}

    strip_kws.setdefault('alpha', 0.5)
    strip_kws.setdefault('zorder', 0)

    if point_kws is None:
        point_kws = {}

    point_kws.setdefault('color', 'k')
    point_kws.setdefault('join', False)
    point_kws.setdefault('markers', '_')

    g = sns.stripplot(**common_opts, **strip_kws, ax=ax)
    sns.pointplot(ax=g, **common_opts, **point_kws)
    g.axes.set_xlabel(xlabel)
    g.axes.set_ylabel(ylabel)


    if legend:
        h, l = g.get_legend_handles_labels()
        g.legend(h, legend, title='')

    return g