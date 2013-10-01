#!/usr/local/bin/python
# encoding: utf-8
"""
Usage: python hw2.py {-fv} --start=<date> {--end=<date>|--duration=<# days>} 
                     --dataset=(2008|2012) --threshold=<threshold>
                     --study.out=<filename> --study.view=(True|False) --study.compare=<index>
                     --trades.out=<filename> --trades.action=(BUY|SELL) --trades.amount=<# shares> --trades.after=<# days> 

Given a date range and dataset, finds when events affecting a stock in the dataset occurred.
The default event is the actual close of a stock falling below the threshold (i.e., the 
previous day's actual close is greater than or equal to the threshold and current day's actual close
is below)

If study.out is provided, an pdf event study is generated.  If study.view is true, at the end of the
program the pdf is displayed.  study.compare indicates what index to compare the returns to.  Defaults to $SPY

If trades.out is provided, an csv orders file is generated.  On the day the event occurs, <trades.action> oder for <trades.amount> shares
is executed. (<trades.action> defaults to 'BUY' and trades.amount defaults to 100).  And <trades.after> trading days after the event, the
inverse order is done (<trades.after> defaults to 5)

Created by Space on 2013-09-24.
Copyright (c) 2013. All rights reserved.

"""
import csv
import datetime as dt
import getopt
import itertools
import os.path
import sys
from subprocess import call

# QSTK Imports
import QSTK.qstkstudy.EventProfiler as ep
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du
import numpy as np


# CONSTANTS
KNOWN_DATASETS = ( "sp5002008", "sp5002012" )
TRADE_ACTIONS = ("BUY", "SELL")

verbose = False

def debug(*args):
    global verbose
    if verbose:
        for arg in args:
            print arg,
        print


def add_year(date):
    later = date + dt.timedelta(days=365)
    if (date.year % 4 == 3 and date.month > 2) or \
        (date.year % 4 == 0 and date.month <= 2):
        later += dt.timedelta(days=1)
    return later


def ymd(dt_day):
    return [dt_day.year, dt_day.month, dt_day.day]



def find_actual_close_pre_ge_now_lt(threshold, d_data):
    """
    Given a threshold and a dictionary of DataFrames including actual_close data,
    returns an Event Matrix indicating the days a stock fell below threshold
    See get_events for an example of the Event Matrix
    
    """
    debug("finding events")
    df_ge = d_data['actual_close'] >= threshold
    df_found = df_ge.shift().fillna(False) & -df_ge
    debug("found %d events with threshold %f" % (df_found.values.sum(), threshold))
    return df_found
    
def get_events(dt_start, dt_end, dataset, find_events, extra_symbols=['$SPY']):
    """
    Given a date range, the name of a dataset, a predicate for identifying events,
    returns an EventMatrix indicating the days an event occurred for a stock as well
    as the DataFrame of data.
    It optionally takes extra_symbols to add to the dataset 
    
    The Event Matrix is a boolean DataFrame with rows as days and columns as stock.
    e.g.,
        |IBM |GOOG|XOM |MSFT| GS | JP |
    (d1)| 0  | 0  | 1  | 0  | 0  | 1  |
    (d2)| 0  | 1  | 0  | 0  | 0  | 0  |
    (d3)| 1  | 0  | 1  | 0  | 1  | 0  |
    (d4)| 0  | 1  | 0  | 1  | 0  | 0  |
    
    """
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    
    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list(dataset)
    ls_symbols.extend(extra_symbols)
    
    debug("getting data")
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    
    return find_events(d_data), d_data

def output_study(df_events, df_data, out_file, compare='$SPY', look=20):
    # convert to float32 to match expected type for eventprofiler
    df_events = (df_events * 1.0).replace(0, np.NaN)

    debug("Creating study '%s'" % out_file)
    debug("Looking %d days around event and comparing to %s" % (look, compare))
    ep.eventprofiler(df_events, df_data, i_lookback=look, i_lookforward=look,
                s_filename=out_file, b_market_neutral=True, b_errorbars=True,
                s_market_sym=compare)


def output_trades(df_events, out_file, amount=100, action="BUY", after=5):
    inverse_action = TRADE_ACTIONS[(TRADE_ACTIONS.index(action)+1)%2]
    
    with open(out_file, "wb") as out:
        writer = csv.writer(out)
        for ndx in range(df_events.shape[0]):
            row = df_events.iloc[ndx]
            
            for stock in itertools.imap(lambda t: t[0], 
                            itertools.ifilter(lambda t: t[1], row.iteritems())):
                writer.writerow(ymd(df_events.index[ndx]) + [stock,action,amount])
                if ndx + after < df_events.shape[0]:
                    writer.writerow(ymd(df_events.index[ndx+after]) + [stock, inverse_action, amount])


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

    global verbose
    force = False
    dataset = "sp5002012"
    dt_start = None
    dt_end = None
    threshold = 5.0
    study_out = None
    study_options = { 'compare' : '$SPY'}
    trades_out = None
    trades_options = {}
    find_events = None
    
    # parse arguments
    try:
        opts, args = getopt.getopt(argv[1:], "?hvfs:e:d:t:c:", \
                                   ["help","verbose","force",
                                    "start=","end=","duration=",             # date range options
                                    "dataset=","threshold=",                 # find events options
                                    "study.out=","study.view=", "view="
                                        "study.compare=","compare=",
                                        "study.look=",                       # event study options
                                    "trades.out=","trades.after=",
                                        "trades.amount=","trades.action="])  # trade options
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
        elif option in ("--dataset"):
            dataset = None
            for known in KNOWN_DATASETS:
                if known.find(value) != -1:
                    if dataset is None:
                        dataset = known
                    else:
                        raise Usage("dataset string '%s' matches multiple known datasets: '%s' and %s" % (value, dataset, known))
            if dataset is None:
                print "unknown dataset '%s'.  Proceeding anyway" % value
                dataset = value
        elif option in ("-t","--threshold"):
            threshold = float(value)
        elif option == "--study.out":
            study_out = value
            if study_out.find(".") == -1:
                study_out = study_out + ".pdf"
        elif option in ("--study.view", "--view"):
            study_options['view'] = bool(value)
        elif option in ("-c", "--study.compare","--compare"):
            study_options['compare'] = value
        elif option == "--study.look":
            study_options['look'] = int(value)
        elif option == "--trades.out":
            trades_out = value
            if trades_out.find(".") == -1:
                trades_out = trades_out + ".csv"
        elif option in ("--trades.after","--trades.amount"):
            trades_options[option.split('.')[1]] = int(value)
        elif option == "--trades.action":
            value = value.upper()
            for action in TRADE_ACTIONS:
                if action.startswith(value) or value.startswith(action):
                    value = action
                    break
            trades_options['action'] = action
        else:
            raise Usage("unknown option: %s" % option)
    if args:
        raise Usage("unknown arguments: %s" % " ".join(args))
    
    if trades_out is None and study_out is None:
        raise Usage("must specify some output, either trades or study")
    
    if dt_start is None:
        raise Usage("must specify start date")
    if dt_end is None:
        # default to a year
        dt_end = add_year(dt_start)
    if dt_start >= dt_end:
        raise Usage("end date must be greater than start date")
    
    if not force:
        for desc, out in (('study', study_out), ('trades', trades_out)):
            if out and os.path.exists(out):
                raise Usage("%s file '%s' already exists.  Add -f to force overwrite" % (desc, out))
    debug("studying dataset %s from %s to %s" % \
          (dataset, dt_start.strftime("%Y-%m-%d"), dt_end.strftime("%Y-%m-%d")))
    debug("Comparing to %s" % study_options['compare'])

    if find_events is None:
        debug("Looking for previous price >= %f and current price < %f" \
            % (threshold, threshold))
        def find_events(df_data):
            return find_actual_close_pre_ge_now_lt(threshold, df_data)

    df_events, df_data = get_events(dt_start, dt_end, dataset, find_events, extra_symbols=[study_options['compare']])
    if trades_out is not None:
        output_trades(df_events, trades_out, **trades_options)
    if study_out is not None:
        study_view = study_options.pop('view',False)
        output_study(df_events, df_data, study_out, **study_options)
        if study_view:
            print "opening study '%s'" % study_out
            call(["open",study_out])
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


