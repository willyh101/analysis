import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def sem(data, axis):
    return data.std(axis)/np.sqrt(data.shape[axis])

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