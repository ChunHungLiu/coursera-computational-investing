#!/usr/local/bin/python
# encoding: utf-8
"""
hw1.py

Created by Space on 2013-09-24.
Copyright (c) 2013. All rights reserved.
"""
import pdb

import sys
import getopt
import datetime as dt

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np

# main arguments
verbose = False

help_message = '''
Calculate the sharpe ratio of a portfolio of stocks

usage: hw1 --start=<date> {--end=<date>|--duration=<# days>} [<equity>{=<allocation>}...]
'''

def debug(*args):
    global verbose
    if verbose:
        for arg in args:
            print arg,
        print


def get_data(start_date, end_date, equities):
    if start_date >= end_date: raise ValueError("end_date should be after start_date")
    
    dataobj = da.DataAccess('Yahoo', cachestalltime=0)
    
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)
    
    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(start_date, end_date, dt_timeofday)    
    
    ldf_data =dataobj.get_data(ldt_timestamps, equities, 'close')

    # Filling the data for NAN
    ldf_data = ldf_data.fillna(method='ffill')
    ldf_data = ldf_data.fillna(method='bfill')
    ldf_data = ldf_data.fillna(1.0)

    return ldf_data

def normalize_prices(ldf_data):
    n_prices = ldf_data.values
    return n_prices / n_prices[0]


def calculate_returns(n_prices, allocation):

    p_closes = np.sum(n_prices * allocation, axis=1)
    daily_rets = p_closes.copy()
  
    # weirdly includes first day with return of 0
    tsu.returnize0(daily_rets)  
    # alternatively
    # daily_rets = p_closes / np.array([p_closes[0]] + p_closes[:-1].tolist()) - 1

    avg_daily_ret = np.mean(daily_rets)
    volatility = np.std(daily_rets)
    # note that the square root of 252 is used to convert from daily data to annualized data and 252 is
    # determined by the size of each sample (a day), NOT the number of samples
    sharpe = np.sqrt(252) * avg_daily_ret / volatility

    return { 'sharpe': sharpe,
             'average_daily_return': avg_daily_ret,
             'cumulative_return': p_closes[-1]/p_closes[0] - 1,
             'volatility': volatility,
             }

def get_date_details(ldf_data):
    dates = ldf_data.axes[0]
    return {'num_trading_days': len(dates),
            'start_date': dates[0],
            'end_date': dates[-1],
            }


def merge(d1, d2):
    new_d = d1.copy()
    new_d.update(d2)
    return new_d
    
    
def simulate_portfolio(start_date, end_date, portfolio):
    """
    simulate the performance of a selection of stocks from a given start_date 
    to end_date. portfolio is a dictionary where the keys are stocks and 
    values are the percentage allocation of the stock in the portfolio
    
    Returns a dictionary with 
        start_date: first trading day in interval
        end_date: last trading day in interval
        start_value: value on close of first day
        end_value: value on close of last day
    
    """
    if not portfolio: raise ValueError("portfolio should be non-empty")
    if sum(portfolio.values()) != 1: raise ValueError("portfolio allocation should be weighted to sum to 1.0")
    
    ldf_data = get_data(start_date, end_date, portfolio.keys())
    return merge(calculate_returns(normalize_prices(ldf_data), portfolio.values()),
                 get_date_details(ldf_data))


def bucketize(count, num_buckets):
    """
    return a generator of arrays of size num_buckets with all the different
    permutations of count objects in the buckets.  The returned array should
    be considered immutable as it is reused within the generator
    
    """
    ar = [0] * num_buckets
    remaining = count
    ndx = 0
    while (ar[0] >= 0):
        ar[ndx] = remaining
        yield ar
        
        # if we're at the end of the array, just go backwards
        # until there's a bucket with something to take from
        if ndx + 1 >= len(ar):
            ar[ndx] = 0
            while ndx > 0:
                ndx = ndx - 1
                if ar[ndx] != 0:
                    break
        else:
            remaining = 0

        ar[ndx] = ar[ndx]-1
        remaining = remaining + 1
        ndx = ndx + 1
    
    
def optimize_portfolio(start_date, end_date, equities, step=0.1):
    """
    optimize a portfolio made up of equities based on the best 
    sharpe ratio evaluated on trading days between start_date and end_date
    It uses step for the smallest non-zero portion of the portfolio in a 
    single stock
    
    """
    assert np.modf(1.0/step)[0] == 0, \
           "step %f must integrally divide 1.0" % step
    ldf_data = get_data(start_date, end_date, equities)
    n_prices = normalize_prices(ldf_data)
    
    best_result = None
    best_alloc = None
    num_permutations = 0
    for alloc in bucketize(int(1.0/step), len(equities)):
        alloc = np.array(alloc) * step
        result = calculate_returns(n_prices, alloc)
        num_permutations = num_permutations + 1
        debug(num_permutations, ':', alloc, ':', result['sharpe'])
        if best_result is None or result['sharpe'] > best_result['sharpe']:
            debug('\t**new best!', result['sharpe'], '>', 
                  (best_result is None and 'nothing' or best_result['sharpe']))
            best_result = result
            best_alloc = alloc

    return (dict(zip(equities,best_alloc)),
            merge(best_result, get_date_details(ldf_data)))
            
            
def add_year(date):
    later = date + dt.timedelta(days=365)
    if (date.year % 4 == 3 and date.month > 2) or \
        (date.year % 4 == 0 and date.month <= 2):
        later += dt.timedelta(days=1)
    return later

######################
# main
######################

class Usage(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    portfolio = {}
    start_date = None
    end_date = None
    optimize = False
    global verbose
    
    # parse arguments
    try:
        opts, args = getopt.getopt(argv[1:], "hvos:e:d:", \
                                   ["help","optimize",
                                    "start=","end=","duration="])
    except getopt.error, msg:
        raise Usage(msg)
    
    # option processing
    for option, value in opts:
        if option == "-v":
            verbose = True
        elif option in ("-h", "--help"):
            raise Usage(help_message)
        elif option in ("-o", "--optimize"):
            optimize = True
        elif option in ("-s", "--start"):
            start_date = dt.datetime.strptime(value, "%Y-%m-%d")
        elif option in ("-e", "--end"):
            if end_date is None:
                end_date = dt.datetime.strptime(value, "%Y-%m-%d")
            else: raise Usage("cannot specify both end and duration")
        elif option in ("-d", "--duration"):
            if end_date is None:
                end_date = start_date + dt.timedelta(days=int(value))
            else: raise Usage("cannot specify both end and duration")
    if start_date is None:
        raise Usage("must specify start date")
    if end_date is None:
        # default to a year
        end_date = add_year(start_date)
    if start_date >= end_date:
        raise Usage("end date must be greater than start date")
    if not args:
        raise Usage("must specify stock")
    
    if optimize:
        (portfolio, result) = optimize_portfolio(start_date, end_date, args)
    else:
        # if we're not optimizing, pull out possible allocations from
        # argument list
        total = 0
        for arg in args:
            stock_allocation = arg.split("=",1)
            allocation = len(stock_allocation) == 1 and 1.0/len(args) or \
                float(stock_allocation[1])
            portfolio[stock_allocation[0]] = allocation
            total += allocation
        if total != 1:
            print "Normalizing allocation from %f to 1.0" % total
            for (key,val) in portfolio.entries():
                portfolio[key] = val / total
        # now simulate the portfolio for the results            
        result = simulate_portfolio(start_date, end_date, portfolio)
    
    print "From %s to %s" \
        % tuple(date.strftime("%Y-%m-%d") for date in (start_date, end_date))
    print "\t(%d trading days)" % result['num_trading_days']
    
    print "%sPortfolio:" % (optimize and "Optimized " or "")
    for equity in sorted(portfolio.keys()):
        print "\t%0.2f %s" % (portfolio[equity],equity)
    print
     
    for prefix in ('cumulative','average_daily'):
        print "%s return: " % prefix.replace('_',' '), \
              result[prefix + '_return']
    print "volatility: ", result['volatility']
    print "sharpe ratio: ", result['sharpe']
    # for key_value in result.items(): 
    #     print "%s: %s" % key_value
    return 0


if __name__ == "__main__":
    ret = 1
    try:
        ret = main()
    except Usage, err:
        print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        print >> sys.stderr, "\t for help use --help"
        ret = 2
    sys.exit(ret)




