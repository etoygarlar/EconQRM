# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:34:17 2017

@author: etr430
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import os
import scipy.optimize as opt
from scipy.stats import norm,t
import sys
from sympy import *
import datetime


def gamma(data, h=0):
    return (data.Ret * data.Ret.shift(periods=h)).dropna().sum()


def realized_variance(data):
    return gamma(data)


def Bipower(data):
    return np.pi / 2 * gamma(data, 1)


def compute_rk(data, H=2):
    kernel = lambda h: np.exp(-0.5 * np.square(h)) / np.sqrt(2 * np.pi)
    realized_kernel = gamma(data)
    for h in range(-H+1, H):
        realized_kernel += kernel(h / H) * gamma(data, h)
    return realized_kernel


def FreqIntradayVar(dataDate, Freq):
    varIntraDayFreq = []
    for i in Freq:
        df = dataDate.Ret[::i]
        varIntraDayFreq.append(np.sum(np.square(df.values)))
    return varIntraDayFreq


def plot_result(df, X, tag):
    f = plt.figure(figsize=(10, 10))
    plt.plot(X,df)
    plt.xlabel('Date')
    plt.ylabel(tag)
    # save plot to file
    plt.savefig('%s.png' % tag)


def plot_result_date_series(s, tag):
    f = plt.figure(figsize=(10, 10))
    s.plot()
    plt.ylabel(tag)
    plt.savefig('%s.png' % tag)


def writeRes(file_name,tag,res):
    with open(file_name, 'a') as f:
        f.write('\n\nRESULT FOR (%s) \n' % tag)
        f.write(str(res))


def main():
    data = pd.read_csv('sp5_1m_2010.csv', usecols=['Date', 'Time', 'C'])
    data = data[data.Time <= 1600]
    data = data[data.Time >= 930]

    data.Date = pd.to_datetime(data.Date)

    data['LogPrice'] = np.log(data.C)
    data['Ret'] = data.LogPrice.diff()
    data = data.dropna(how='any')

    pickDate = pd.to_datetime('2010-05-06', format='%Y-%m-%d')
    dataDate = data[data.Date == pickDate]

    frequencies = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 120, 194]

    intraday_realized_var = FreqIntradayVar(dataDate, frequencies)
    print('The picked date realized variance : %s' % intraday_realized_var)
    plot_result(intraday_realized_var, frequencies, 'Picked date variance')

    data5min = data[::5]

    realized_5min_var = data5min.groupby('Date').apply(realized_variance)
    plot_result_date_series(realized_5min_var, 'All 5min data variance')
    writeRes('RealizedVariance5min', 'Realized Variance of 5min Data', realized_5min_var)

    bipower_5min_var = data5min.groupby('Date').apply(Bipower)
    print('Bipower variance of 5min data : %s \n' %bipower_5min_var)
    plot_result_date_series(bipower_5min_var, 'Bipower variance of all 5min data')
    writeRes('BipowerVariance5min', 'Bipower Variance of 5min Data', bipower_5min_var)

    realized_1min_var = data.groupby('Date').apply(realized_variance)
    print('The realized variance of 1min data: %s' % realized_1min_var)
    plot_result_date_series(realized_1min_var, 'All data variance')
    writeRes('RealizedVariance1min', 'Realized Variance of 1min Data', realized_1min_var)

    bipower_1min_var = data.groupby('Date').apply(Bipower)
    print('Bipower variance of 1min data : %s \n' % bipower_1min_var)
    plot_result_date_series(bipower_1min_var, 'Bipower variance of all data')
    writeRes('BipowerVariance1min','Bipower Variance of 1min Data', bipower_1min_var)

    realized_kernel_5min = data5min.groupby('Date').apply(compute_rk)
    print('Realised Kernel variance of 5min data with Gaussian : %s \n' % realized_kernel_5min)
    plot_result_date_series(realized_kernel_5min, 'Realised Kernel variance of 5 min data with Gaussian')
    writeRes('RealizedKernel5min', 'Realised Kernel  of 5min Data', realized_kernel_5min)

    realized_kernel_1min = data.groupby('Date').apply(compute_rk)
    print('Realised Kernel variance of 1min data with Gaussian : %s \n' % realized_kernel_1min)
    plot_result_date_series(realized_kernel_1min, 'Realised Kernel variance of 1 min data with Gaussian')
    writeRes('RealizedKernel1min', 'Realised Kernel  of 1min Data', realized_kernel_1min)

    plt.figure(1)
    ax = plt.subplot(211)
    plt.title('5 min data')
    df5min = pd.DataFrame()
    df5min['RV'] = realized_5min_var
    df5min['BV'] = bipower_5min_var
    df5min['RK'] = realized_kernel_5min
    df5min.plot(ax=ax)

    # plt.plot(realized_5min_var, 'r', bipower_5min_var, 'b', realized_kernel_5min, 'g')
    ax = plt.subplot(212)
    plt.title('1 min data')
    df1min = pd.DataFrame()
    df1min['RV'] = realized_1min_var
    df1min['BV'] = bipower_1min_var
    df1min['RK'] = realized_kernel_1min
    df1min.plot(ax=ax)
    # plt.plot(realized_1min_var, 'r', bipower_1min_var, 'b', realized_kernel_1min, 'g')
    plt.show()
    plt.savefig('Comparison.png')


    ######## Plot single day RV
    plt.figure(0)
    plt.plot(frequencies, FreqIntradayVar(dataDate, frequencies))
    plt.title('Chosen Date variance with different frequencies')
    plt.xlabel('i-minute Frequency')
    plt.ylabel('RV')
    plt.show()
    plt.savefig('Frequencies for %s.png' % pickDate)






if __name__ == "__main__":
    main()
