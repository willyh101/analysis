from pathlib import Path
import time
import functools
import os

def make_results_folder(root, mouse, date, folder_name='results', chdir=True):
    """
    Creates a folder named 'results' for the mouse at root/mouse/date/results. Optionally changes 
    the working directory to the results folder. Can specify a folder name other than results or
    set as empty string.

    Args:
        root (str): root path or folder to save into
        mouse (str): name of mouse
        date (str): date of experiment
        folder_name (str): name of folder to save to. leave empty to save to 
                           root/mouse/name. Defaults to 'results'.
        chdir (bool, optional): whether to change the cwd to results folder. Defaults to True.

    Returns:
        pathlib.Path: path to results folder
    """
    results = Path(root, mouse, date, folder_name)
    results.mkdir(exist_ok=True, parents=True)
    if chdir:
        os.chdir(results)
        print(f'Set cwd to: {os.getcwd()}')
    return results

def tic():
    """Records the time in highest resolution possible for timing code."""
    return time.perf_counter()

def toc(tic):
    """Returns the time since 'tic' was called."""
    return time.perf_counter() - tic

def ptoc(tic, start_string='Time elapsed:', end_string='s'):
    """
    Print a default or custom print statement with elapsed time. Both the start_string
    and end_string can be customized. Autoformats with single space between start, time, 
    stop. Returns the time elapsed.

    Format -> 'start_string' + 'elapsed time in seconds' + 'end_string'.
    Default -> start_string = 'Time elapsed:', end_string = 's'.
    """
    t = toc(tic)
    pstring = ' '.join([start_string, f'{t:.4f}', end_string])
    print(pstring)
    return t

def ptoc_min(tic, start_string='Time elapsed:', end_string='min'):
    """See ptoc. Modified for long running processes."""
    t = toc(tic)
    pstring = ' '.join([start_string, f'{t/60:.2f}', end_string])
    print(pstring)
    return t

def tictoc(func):
    """Prints the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f'<{func.__module__}.{func.__name__}> done in {run_time:.3f}s')
        return value
    return wrapper_timer

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug

def verifyrun(func):
    """Prints whether the decorated function ran."""
    @functools.wraps(func)
    def wrapper_verifyrun(*args, **kwargs):
        print(f'Ran {func.__name__!r} from {func.__module__}.')
        value = func(*args, **kwargs)
        return value
    return wrapper_verifyrun

def replace_tup_ix(tup, ix, val):
    return tup[:ix] + (val,) + tup[ix+1:]

def nbsetup():
    try:
        if __IPYTHON__:
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
            get_ipython().magic("config InlineBackend.figure_format = 'retina'")
    except NameError:
        pass
    
    import seaborn as sns
    import matplotlib as mpl
    import pandas as pd
    
    sns.set_style('ticks',{'axes.spines.right': False, 'axes.spines.top': False}) # removes annoying top and right axis
    sns.set_context('notebook') # can change to paper, poster, talk, notebook

    pd.set_option('display.max_columns',10) # limits printing of dataframes

    mpl.rcParams['savefig.dpi'] = 600 # default resolution for saving images in matplotlib
    mpl.rcParams['savefig.format'] = 'pdf' # defaults to png for saved images (SVG is best, however)
    mpl.rcParams['savefig.bbox'] = 'tight' # so saved graphics don't get chopped
    mpl.rcParams['image.cmap'] = 'viridis'
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.transparent'] = True
    mpl.rcParams['pdf.fonttype'] = 42

def flatten(t):
    return [item for sublist in t for item in sublist]