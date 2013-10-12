#!/usr/local/bin/python
# encoding: utf-8
"""
Usage: python bollinger.py {-f} {-v} --lookback=<lookback> --start=<start_date> {--end=<end date>} 
                           {-o <csv output>} {-g <graph file>} {-view} EQUITY...
                           
Generates the bollinger bands for a given set of equities.
By default, it will print the bollinger bands to standard output.
If an output is provided, it will write the bollinger bands at in a csv file in
the following format:

2008, 12, 3, 50.43, 52.43, 54.43

If a graph file is provided, it will plot the graphs of the stock price and bollinger bands
over the given date period.  If -view is specified, it will open the pdf file after it
has been created.

If multiple equities are provided and output files are specified, the output files will be
prefixed with the stock symbol.

Created by Space on 2013-10-11.
Copyright (c) 2013. All rights reserved.
"""
import pdb

import sys
import getopt
import datetime as dt
import os.path
import csv
from subprocess import call

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# QSTK Imports
import QSTK.qstkstudy.EventProfiler as ep
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du

verbose = False

def debug(*args):
    global verbose
    if verbose:
        for arg in args:
            print arg,
        print

def ternary(cond, t, f):
    if cond:
        return t
    else:
        return f

def get_data(dt_start, dt_end, symbols, lookback=0):
    """
    Given a date range, return the adjusted_close price for the given symbols
    If lookback is specified, it will move the start_date back that many trading days
    """
    assert(lookback >= 0)
    
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    if lookback >= 0:
        lookback_timestamps = du.getNYSEdays(dt_start - dt.timedelta(days=lookback*2),dt_start, dt.timedelta(hours=16))
        if ldt_timestamps[0] == lookback_timestamps[-1]:
            debug("contains the end time")
            lookback_timestamps.pop() 
        ldt_timestamps = lookback_timestamps[-(lookback-1):] + ldt_timestamps

    dataobj = da.DataAccess('Yahoo')
    debug("getting data")
    ldf_data = dataobj.get_data(ldt_timestamps, symbols, 'close')

    ldf_data = ldf_data.fillna(method='ffill')
    ldf_data = ldf_data.fillna(method='bfill')
    ldf_data = ldf_data.fillna(1.0)

    return ldf_data

def get_rolling_stats(ldf_data, lookback):
    pnl_stats = pd.Panel({'price': ldf_data[lookback-1:],
                          'mean': pd.rolling_mean(ldf_data,lookback)[lookback-1:],
                          'std': pd.rolling_std(ldf_data,lookback)[lookback-1:]})
    # dict doesn't guarantee order of axes, so reindex since we use order-dependence
    # when reading rows into tuple
    pnl_stats = pnl_stats.reindex_axis(('price','mean','std'), axis=0)
    # reorder axes to stocks,dates,stat
    return pnl_stats.transpose(2,1,0)

def raise_usage_if_exists(filename):
    if os.path.exists(filename):
        raise Usage("file '%s' already exists.  Add -f to force overwrite" \
                    % filename)
    return filename

def bollinger_value(price, mean, std):
    debug("price = %.2f, mean=%.2f std=%.2f" % (price, mean, std))
    return (price - mean)/std

def bollinger(price, mean, std):
    return [mean-std, mean, mean+std, bollinger_value(price, mean, std)]
    
def write_csv_bollinger(out_file, df):
    debug("Writing bollinger data to %s" % out_file)
    with open(out_file, "rb") as out:
        csvwriter = csv.writer(out)
        for (dt_day, row) in df.iterrows():
            
            csvwriter.writerow([dt_day.year, dt_day.month, dt_day.day] + bollinger(*row))

def graph_bollinger(out_file, title, df):
    debug("Plotting price and bollinger lines in %s" % out_file)
    plt.clf()
    plt.plot(df.index, df['price'], "b-", label='price', linewidth=2)
    means = df['mean']
    stds = df['std']
    plt.plot(df.index, means, "k-", label='mean', linewidth=4)
    plt.plot(df.index, means-stds, "r-", label='bollinger band', linewidth=1)
    plt.plot(df.index, means+stds, "r-", label='bollinger band', linewidth=1)
    plt.title(title)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.xticks(rotation=-90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, format='pdf')
    
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

    dt_start = None
    dt_end = None
    equities = []
    graph_file = None
    out_file = None
    force = False
    lookback = None
    view = False
    global verbose

    # parse arguments
    try:
        opts, args = getopt.getopt(argv[1:], "?hvfs:e:d:l:o:g:",
                                   ["help","force",
                                    "start=","end=","duration=", 
                                    "lookback=","out=", "graph=",
                                    "view", "view="]);
    except getopt.error, msg:
        raise Usage(msg)

    # option processing
    for option, value in opts:
        if option in ("-?", "-h", "--help"):
            raise Usage(__doc__)
        elif option in ("-v","--verbose"):
            verbose = True
        elif option in ("-f","--force"):
            force = True
        elif option in ("-s", "--start"):
            dt_start = dt.datetime.strptime(value, "%Y-%m-%d")
        elif option in ("-e", "--end"):
            if dt_end is None:
                dt_end = dt.datetime.strptime(value, "%Y-%m-%d")
            else: raise Usage("cannot specify both end and duration")
        elif option in ("-d", "--duration"):
            if dt_end is None:
                dt_end = dt_start + dt.timedelta(days=int(value))
            else: raise Usage("cannot specify both end and duration")
        elif option in ("-o","--out"):
            out_file = value
            if out_file.fine(".") == -1:
                out_file += ".csv"
        elif option in ("-g","--graph"):
            graph_file = value
            if graph_file.find(".") == -1:
                graph_file += ".pdf"
        elif option in ("-l","--lookback"):
            lookback = int(value)
        elif option in ("--view"):
            view = True
        elif option in ("--view="):
            view = bool(value)
        else:
            raise Usage("unknown option: %s" % option)
    # remaining arguments are equities
    equities = args

    if not equities:
        raise Usage("must specify at least 1 equity!")
    if not lookback:
        raise Usage("must specify lookback")
    if not dt_start:
        raise Usage("must specify the start of bollinger data")    

    if not dt_end:
        dt_end = dt_start + dt.timedelta(days=1)
    check_filename = ternary(force, (lambda filename : filename),
                                    raise_usage_if_exists)
    ldf_data = get_data(dt_start, dt_end, equities, lookback=lookback)
    pnl_stats = get_rolling_stats(ldf_data, lookback)
    
    gen_filename = ternary(pnl_stats.shape[0] == 1,
                           (lambda _, filename: filename),
                           (lambda sym, filename: ".".join((sym, filename))))
    if out_file:
        for (sym, df) in pnl_stats.iteritems():
            write_csv_bollinger(check_filename(gen_filename(sym, out_file)),
                                df)
    else:
        for (sym, df) in pnl_stats.iteritems():
            prices = ldf_data[sym]
            print "%s:" % sym
            for (dt_day, row) in df.iterrows():
                print "%s: low=%.2f, mean=%.2f, high=%.2f, bollinger value=%.2f" \
                        % tuple([dt_day.strftime("%Y-%m-%d")] + bollinger(*row))
                          
    
    if graph_file:
        for (sym, df) in pnl_stats.iteritems():
            sym_graph_file = check_filename(gen_filename(sym, graph_file))
            graph_bollinger(sym_graph_file,
                            sym, df)
            if view:
                debug("opening %s" % sym_graph_file)
                call(["open",sym_graph_file])

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



