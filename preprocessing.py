from scipy.signal import savgol_filter
from peakutils import baseline
from silx.math.fit import snip1d
import numpy as np
import pandas as pd


def smoothing(x, window_length, polyorder, deriv=0, mode='interp'):
    """Apply Savitzky-Golay filter to smooth an array. This is a wrapper
    around scipy.signal.savgol_filter.
    """
    return savgol_filter(x, window_length=window_length, polyorder=polyorder,
                         deriv=deriv, mode=mode)


def background(x, n=2, type='poly', max_it=100, tol=0.001):
    """Apply baseline reduction using baseline polynomial fitting or SNIP
    algorithm. This is a wrapper around peakutils.baseline and
    silx.math.fit.snip1d.
    """
    if type not in ['poly', 'snip']:
        raise ValueError("type must be 'poly' or 'snip'")
    elif type == 'poly':
        return baseline(x, deg=n, max_it=max_it, tol=tol)
    else:
        return snip1d(x, snip_width=n)


def index(x, n):
    '''Returns the index of the dataframe's or numpy array's first-column
    value nearest to the value entered.
    '''
    if type(x) is pd.core.frame.DataFrame:
        return (x[x.columns[0]] - n).abs().argsort()[0]
    elif type(x) is np.ndarray:
        return np.abs(x[:, 0] - n).argsort()[0]


def cut(x, start, end):
    """Cut spectrum based on wavelength.
    """
    return x[index(x, start):index(x, end)]


def icut(x, start, end):
    """Cut spectrum based on index.
    """
    return x[start:end]
