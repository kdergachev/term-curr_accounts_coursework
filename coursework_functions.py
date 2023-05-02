import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels as sm
import statsmodels.api as sma
import statsmodels.tsa.api as smt

from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import welch

from itertools import accumulate

import matplotlib.pyplot as plt

from pygam import *
from pygam.utils import combine

from tqdm import tqdm


class OwnCVIndex:
    
    """Time series CV class. kinddict dictates what kinds of 
    subdivision are used: [{0, 1}, {0, 1}] - [train (fixed, sliding), 
                                              test (fixed, sliding)]"""
    
    
    def __init__(self, dataset, train_size, test_size, number_of_runs=None):
        
        """Setup for generator. If number of runs is None slide by 1 each new 
        fold o/w divide maximum amount of runs by number of runs (floored)
        minus one to get step size. dataset - actual dataset which is 
        split with observations on first axis"""
        
        
        self.train_size = train_size
        self.test_size = test_size
        self.size = len(dataset)
        
        taken = train_size + test_size
        self.max_runs = self.size - taken + 1
        if number_of_runs is not None:
            self.increment = self.max_runs//(number_of_runs - 1)
            self.n = number_of_runs
        else:
            self.increment = 1
            self.n = self.max_runs
        self.position = 0
        
    
    kinddict = {'sliding-win': np.array([1, 1, 1, 1]),
                'sliding-test': np.array([0, 0, 1, 1]),
                'expanding-win': np.array([0, 1, 1, 1]),
                'reducing-test': np.array([0, 1, 1, 0])}
    
    
    def generate_idx(self, kind, lag):
        
        """Generator method to get indices of train, test and init data for 
        each fold. Lag is used to give indices for the model reinitialization"""
        
        
        start = np.array([self.position, 
                          self.position + self.train_size, 
                          self.position + self.train_size, 
                          self.position + self.train_size + self.test_size])
        
        kind = OwnCVIndex.kinddict[kind]
        for i in range(self.n):
            diff = kind * self.increment * i
            temp = start + diff
            train = np.arange(temp[0], temp[1])
            test = np.arange(temp[2], temp[3])
            init = np.arange(temp[2] - lag, temp[2])
            yield train, test, init




def add_lags(table, lags, columns=None):
    
    """Add lags of selected columns to the dataframe. table - dataframe to 
    process, lags - lag order list, columns - names of columns to add lags to.
    returns dataframe with new N(lags)*N(columns) columns named L_.colname"""
    
    if columns is None:
        columns = table.columns
    res = table.copy()
    for i in lags:
        temp = res[columns].shift(i)
        temp.columns = pd.Index(['L' + str(i) + '.' + j for j in columns])
        res = pd.concat([res, temp], axis=1)
    return res


def test_stationarity(table, columns=None):
    
    """ACF test of multiple columns in a table at the same time"""
    
    if columns is None:
        columns=table.columns
    results = {}
    for i in columns:
        results[i] = (smt.adf(table[i].dropna())[1], smt.adf(table[i].dropna())[3])
    return results



def periodogram(series, freq=None):
    
    """Plot periodogram of series using welch method, print 15 peaks calculated 
       as 15 points with greatest ratio of given value divided by average of 
       this point, three previous points and three next points"""

    if freq is None:
        freq = len(series)
    
    fx, fy = welch(series, freq)
    f = pd.Series(fy, index=fx)
    #fy = rfft(np.array(series))[1:]
    #fx = rfftfreq(len(series), freq)[1:]
    
    plt.grid()
    #f.plot()
    plt.plot(f.index, f)
    plt.show()
    print(f.rolling(7).apply(lambda x: x.iloc[4]/np.mean(x)).sort_values(ascending=False).iloc[:15])


def harmonics_search(data, sins=(), coss=(), outputs=True):
    
    """Search harmonics of given dataset and return sin and cos series. 
    Input data, wavelength of harmonics, and boolean for showing tests/only 
    generating sin/cos series"""
    
    # acos + bsin = sqrt(a**2 + b**2)cos( + arctan(-b/a) + I(a < 0)pi)
    
    if not sins and not coss:
        periodogram(data)
        return None
    
    harmonics = pd.DataFrame(index=data.index)
    for i in sins:      
            harmonics = pd.concat([harmonics, pd.Series(np.sin(2*np.pi*(1/i)*np.arange(0, len(data))), 
                                                        index=data.index, name='s'+str(i))], axis=1)
    for i in coss:
            harmonics = pd.concat([harmonics, pd.Series(np.cos(2*np.pi*(1/i)*np.arange(0, len(data))), 
                                                        index=data.index, name='c'+str(i))], axis=1)
    
    mod = sma.OLS(data, sma.add_constant(harmonics)).fit()
    if outputs:
        periodogram(mod.resid)
        periodogram(data)
        print(mod.summary())
        
        hnames = harmonics.columns
        hnames = pd.Series(hnames).apply(lambda x: x[1:]).value_counts()
        hnames = hnames[hnames== 2].index
        jointtests = pd.DataFrame(columns=['Fstat', 'p-val'], index=pd.Series(hnames, name='H0').apply(lambda x: f"c{x}=s{x}=0"))
        for i in hnames:
            jointtests.loc[f"c{i}=s{i}=0", 'p-val'] = mod.f_test(f"(s{i} = 0), (c{i} = 0)").pvalue
            jointtests.loc[f"c{i}=s{i}=0", 'Fstat'] = mod.f_test(f"(s{i} = 0), (c{i} = 0)").fvalue
        print(jointtests)
        
        mod.resid.plot()
    return harmonics
    


def order_search(data, lags=40, **kwargs):
    
    """Get model summary and acf, pacf and spectrogram of residuals for a 
    given series. Data endogenous in SARIMAX, lags - maxlag shown on acf/pacf,
    all else is treated as input to SARIMAX"""
    
    assert(~data.isna().iloc[0])
    model = smt.SARIMAX(data, **kwargs).fit()
    upto = model.loglikelihood_burn
    print(model.resid.iloc[upto:])
    plot_acf(model.resid.iloc[upto:], zero=False, auto_ylims=True, lags=lags)
    plt.show()
    plot_pacf(model.resid.iloc[upto:], zero=False, auto_ylims=True, lags=lags)
    plt.show()
    print(model.resid.iloc[upto:])
    print(model.summary())
    model.resid.iloc[upto:].plot()
    plt.show()
    periodogram(model.resid.iloc[upto:])


def exp_smoothing(data, alpha):
    """Simple exponential smoothing function"""
    
    
    res = list(accumulate(data, lambda x, y: alpha*x + (1-alpha)*y))
    if isinstance(data, pd.Series):
        res = pd.Series(res, index=data.index, name=data.name)
    if isinstance(data, np.ndarray):
        res = np.array(res)
    return res


def errors_to_stats(errors, cs=True):
    """Misc function used to convert arrays of errors to mean MAE, RMSE and MAE_L"""

    if cs:
        f = lambda x: np.cumsum(x, axis=1)
    else:
        f = lambda x: np.array(x)
    
    resd = {'MAE': np.abs(f(errors)).mean(axis=1),
            'RMSE': np.sqrt((f(errors)**2).mean(axis=1)),
            'last_err': np.abs(f(errors)[:, -1])}
    return resd



