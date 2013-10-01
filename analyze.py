#!/usr/local/bin/python
# encoding: utf-8
"""
Usage: python analyze.py {-f} {-v} <portfolio file> <output graph>

Takes a file listing the daily value of a portfolio and analyzes it in comparison to SPY$

The csv file for the portfolio should look like:

2008, 12, 3, 1000000
2008, 12, 4, 1000010
2008, 12, 5, 1000250

If an output file is provided, a graph will be generated plotting portfolio and SPY's daily values

Created by Space on 2013-10-01.
Copyright (c) 2013. All rights reserved.
"""
import pdb

import sys
import copy
import getopt
import datetime as dt
import os.path
from subprocess import call
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkstudy.EventProfiler as ep

verbose = False

def debug(*args):
    global verbose
    if verbose:
        for arg in args:
            print arg,
        print


def add_to_range(dt_range, dt):
    if dt < dt_range[0]:
        dt_range[0] = dt
    if dt > dt_range[1]:
        dt_range[1] = dt

def get_data(dt_range, equities):
    dataobj = da.DataAccess('Yahoo') # , cachestalltime=0)
    
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)
    
    ldf_data =dataobj.get_data([d + dt_timeofday for d in dt_range], equities, 'close')

    # Filling the data for NAN
    ldf_data = ldf_data.fillna(method='ffill')
    ldf_data = ldf_data.fillna(method='bfill')
    ldf_data = ldf_data.fillna(1.0)

    return ldf_data


def get_stats(values):
    """
    generate statistics for these market returns.  Returns a dictionary with statistics in it.
    Statistics returned are: 'sharpe','volatility','avg_daily_return','total_return'
    
    """
    rets = values.copy()
    tsu.returnize0(rets)
    
    avg = np.mean(rets)
    vol = np.std(rets)
    
    # note that the square root of 252 is used to convert from daily data to annualized data and 252 is
    # determined by the size of each sample (a day), NOT the number of sample
    return { 'sharpe': np.sqrt(252)*avg/vol,
             'volatility': vol,
             'avg_daily_return': avg,
             'total_return': values[-1]/values[0] - 1 }
    
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
        
    portfolio_file = None
    out_file = None
    compare = "$SPX"
    force = False
    global verbose
    
    # parse arguments
    try:
        opts, args = getopt.getopt(argv[1:], "?hvfp:o:c:", \
                                   ["help","force",
                                    "portfolio=","out=","compare="])
    except getopt.error, msg:
        raise Usage(msg)
    
    # option processing
    for option, value in opts:
        if option == "-v":
            verbose = True
        elif option in ("-h", "-?", "--help"):
            raise Usage(__doc__)
        elif option in ("-f","--force"):
            force = True
        elif option in ("-p","--portfolio"):
            portfolio_file = value
        elif option in ("-o","--out"):
            out_file = value
        elif option in ("-c","-compare"):
            compare = value
        else:
            raise Usage("unknown option: %s" % option)
    try:
        if portfolio_file is None:
            portfolio_file = args.pop(0)
        if out_file is None and args:
            out_file = args.pop(0)
    except IndexError:
        raise Usage("expected portfolio fil arguments")
    
    if not os.path.exists(portfolio_file) and portfolio_file.find(".") == -1:
        portfolio_file = portfolio_file + ".csv"
    if out_file:
        if out_file.find(".") == -1:
            out_file = out_file + ".pdf"
        if not force and os.path.exists(out_file):
            raise Usage("output file '%s' already exists. Add -f to force overwrite" % out_file)

    debug("Reading portfolio from %s" % portfolio_file)
    debug("Comparing portfolio to %s" % compare)
    debug()

    df_data = pd.io.parsers.read_csv(portfolio_file, parse_dates=[[0,1,2]], index_col = 0, header = None)
    df_data.columns = ['Portfolio']
    ndx = df_data.index
    debug("Read in %d rows of portfolio values from %s to %s" % tuple([len(ndx)] + [d.strftime("%Y-%m-%d") for d in (ndx[0], ndx[-1])]))
    # we're going to assert these are sorted by date
    dt_range = df_data.index 
    assert not [i for i in range(1,len(dt_range)) if dt_range[i-1] >= dt_range[i] ], "dates out of order"
    
    df_compare = get_data(dt_range, [compare])
    ndx = df_compare.index
    debug("Read in %d rows of %s values from %s to %s" % tuple([len(ndx), compare] + [d.strftime("%Y-%m-%d") for d in (ndx[0], ndx[-1])]))

    # make sure indices are the same before concating so that they line up
    df_compare.index = df_data.index
    df_data = pd.concat([df_data,df_compare], axis=1)
    stats = [get_stats(df_data[col]) for col in df_data.columns]

    print "Portfolio started at $%.2f on %s" % (df_data['Portfolio'][0], df_data.index[0].strftime("%Y-%m-%d"))
    print "Portfolio ended at $%.2f on %s" % (df_data['Portfolio'][-1], df_data.index[-1].strftime("%Y-%m-%d"))
    
    print "%-20s" % "", "\t".join("%-15s" % c for c in df_data.columns)
    for k in sorted(stats[0].keys()):
        print "%-20s" % k,
        print "\t".join(str(s[k]) for s in stats)
    
    if out_file:
        debug("Plotting returns in %s" % out_file)
        na_prices = df_data.values
        # normalize prices to start at 1 to see relative returns
        na_prices = na_prices / na_prices[0,:]
        
        plt.clf()
        for ndx in range(na_prices.shape[1]):
            plt.plot(df_data.index, na_prices[:,ndx])
        plt.axhline(y=0,color='r')
        plt.legend(df_data.columns)
        plt.ylabel('Daily Returns')
        plt.xlabel('Date')
        plt.savefig(out_file, format='pdf')
        debug("opening %s" % out_file)
        call(["open",out_file])
        
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



