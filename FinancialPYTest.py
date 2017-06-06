# numpy scipy only work with Python 2.7 Interpretor pointing to python 2.7 instead of python 3.6
import numpy as np
import pandas as pd
from pandas_datareader import data, wb
import pandas_datareader.data as web
#import pandas_datareader.data as web2
import matplotlib.pyplot as plt
import datetime
from pandas_datareader._utils import RemoteDataError
import csv
import sys
import os

print ("import sucessful")

S0 = 100
K = 105
T = 1.0
r = 0.05
sigma = 0.2

I = 100000

# valuation algorithm
z = np.random.standard_normal(I)  # pseudorandom numbers
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
# index values at maturity
hT = np.maximum(ST - K, 0)  # inner values at maturity
C0 = np.exp(-r * T) * np.sum(hT) / I  # Monte Carlo estimator

# Result Output
print("value of the European Call Option %5.3f" % C0)

# Get stock information from google
start = datetime.datetime(2000, 1, 3)
end = datetime.datetime(2017, 5, 12)
goog = web.DataReader('amd', 'google', start, end)

# Display the FIRST 5 prices in your range
print(goog.head())
# Display the LAST 5 prices in your range
print(goog.tail())

# Get stock information from Yahoo
#start = datetime.datetime(2016, 1, 3)
#end = datetime.datetime(2017, 5, 12)
ticker = "amd"
stticker = ticker.strip('\n')

#try:
#yah = web.DataReader(stticker, 'yahoo', start, end)
#except RemoteDataError:
#    print("couldn't read the file")

# Display the FIRST 5 prices in your range
#print(yah.head())
# Display the LAST 5 prices in your range
#print(yah.tail())

# goog['Log_Ret'] = np.log(goog['Close'] / goog['Close'].shift(1))
# goog['Volatility'] = pd.rolling_std(goog['Log_Ret'], window=252) * np.sqrt(252)
# %matplotlib inline
# goog[['Close','Volatility']].plot(subplots=True, color=blue, figsize=(8,6))

#Black-Scholes-Merton (1973) Functions

def bsm_call_value(S00, K0, T0, r0, sigma0):
    '''
    
    :param S00: float
                initial stock/index level
    :param K0:  float
                strike price
    :param T0:  float
                maturity date (in year fractions)
    :param r0:  float
                constant risk-free short rate
    :param sigma0: float
                volatility factor in diffusion term
    :return:    float
                present value of the European call option
    '''
    from math import log,sqrt,exp
    from scipy import stats

    S00 = float (S00)
    d1 = (log(S00/K0)+(r+0.5*sigma0**2)*T0)/(sigma0*sqrt(T0))
    d2 = (log(S00 / K0) + (r + 0.5 * sigma0 ** 2) * T0) / (sigma0 * sqrt(T0))
    value=(S00*stats.norm.cdf(d2,0.0,1.0))
    '''
    stats.norm.cdf --> cumulative distribution function for normal distribution
    '''
    return value

# Vega function
def bsm_vega(S0, K, T, r, sigma):
    '''

    :param S0:  float
                initial stock/index level
    :param K:   float
                strike price
    :param T:   float
                maturity date (in year fraction)
    :param r:   float
                constant risk-free short rate
    :param sigma: float
                volatility factor in diffusion term
    :return:
    vega:       float
                partial derivative of BSM formula with respect to sigma, i.e. vega
    '''
    from math import log, sqrt
    from scipy import stats

    S0=float(S0)
    d1=(log(S0/K)+(r+0.5*sigma**2)*T/(sigma*sqrt(T)))
    vega=S0*stats.norm.cdf(d1,0.0,1.0)*sqrt(T)
    return vega

#Implied volatility function
def bsm_call_imp_vol(S0,K,T,r,C0,sigma_est,it=100):
    '''
    
    :param S0:      float
                    initial stock/index level
    :param K:       float
                    strike price
    :param T:       float
                    maturity date (in year fractions)
    :param r:       float
                    constant risk-free short rate
    :param C0:      float
    :param sigma_est: float
                    estimate of impl. volatility
    :param it:      integer
                    number of iterations
    :return: 
    sigma_est:      float
                    numerically estimated implied volatility
    '''
    for i in range(it):
        sigma_est-=((bsm_call_value(s0,K,T,r,sigma_est)-C0)/bsm_vega(S0,K,T,r,sigma_est))
    return sigma_est

