import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

RNG = np.random.default_rng()

def sem(data, axis=0):
    return data.std(axis)/np.sqrt(data.shape[axis])

def _ci(arr, interval=0.95):
    n = len(arr)
    m, se = np.mean(arr), stats.sem(arr)
    h = se * stats.t.ppf((1 + interval) / 2, n-1)
    return m, m-h, m+h

def ci(arr, interval=0.95):
    inter = stats.t.interval(interval, len(arr)-1, loc=np.mean(arr), scale=stats.sem(arr))
    return np.abs(np.mean(arr) - inter[0])

def sumsquares(x, y, func, popt):
    residuals = y - func(x, *popt)
    return np.sum(residuals**2)

def total_sse(y):
    avg_y = np.mean(y)
    squared_err = (y - avg_y)**2
    return np.sum(squared_err)

def mse(x, y, func, popt):
    y_pred = func(x, *popt)
    return mean_squared_error(y, y_pred)

# def rsquared(x, y, func, popt):
#     return 1 - (sumsquares(x, y, func, popt)/total_sse(y))

def rsquared(x, y, func, popt):
    y_pred = func(x, *popt)
    return r2_score(y, y_pred)

def gauss2d(xy, xo, yo, amplitude, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = np.array(xo).astype(float)
    yo = np.array(yo).astype(float) 
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def gauss2d_simple(xy, xo, yo, amplitude, sigma_x, sigma_y):
    x, y = xy
    xo = np.array(xo).astype(float)
    yo = np.array(yo).astype(float)
    a = ((x-xo)**2)/(2*sigma_x**2)
    b = ((y-yo)**2)/(2*sigma_y**2)
    g = amplitude * np.exp(-(a+b))
    return g.ravel()

def von_mises_dan(x, kappa, mu, ht):
    return (np.exp(kappa*np.cos(x-mu)) + ht*np.exp(kappa*np.cos(x-mu-np.pi)))/(np.exp(kappa) + ht*np.exp(-kappa))

def von_mises_ori(x, kappa, mu):
    return stats.vonmises.pdf(x, kappa, mu)

def von_mises_dir(x, kappa, mu, ht):
    return stats.vonmises.pdf(x, kappa, mu) + stats.vonmises.pdf(x, kappa, mu-np.pi)*ht

def von_mises_sym(x, kappa, offset, ampl):
    return ampl * von_mises_dan(x, kappa, 0, 1) + offset

def bootstrap(x, func, nsample=None, nboot=10000, **kwargs):
    if not nsample:
        nsample = x.shape[0]
    sample = []
    for i in range(nboot):
        y = np.random.choice(x, nsample, replace=True)
        val = func(y, **kwargs)
        sample.append(val)
    return np.array(sample)

def bootstrap2d(x: np.ndarray, func, axis, nsample=None, nboot=10000, **kwargs):
    rng = np.random.default_rng()
    if not nsample:
        nsample = x.shape[axis]
    sample = []
    for i in range(nboot):
        y = rng.choice(x, nsample, axis=axis, replace=True)
        val = func(y, axis=axis, **kwargs)
        sample.append(val)
    sample = np.vstack(sample)
    return sample

def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx!=i]) for i in range(n))/float(n)

def fusshle(x,y):
    xy = np.vstack([x,y]).T.copy()
    RNG.shuffle(xy)
    x = xy[:,0]
    y = xy[:,1]
    return x,y

def splitcorr(arr, frac=0.5, nboot=10000):
    """arr: n (trials) x m (frames) for single cell"""
    ccs = []
    for i in range(nboot):
        nchoose = int(arr.shape[0]*frac)
        perm = RNG.permutation(np.arange(arr.shape[0]))
        a1 = arr[perm[:nchoose],:].mean(axis=0)
        a2 = arr[perm[nchoose:],:].mean(axis=0)
        cc = stats.pearsonr(a1,a2)[0]
        ccs.append(cc)
    return np.array(ccs)