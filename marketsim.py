#!/usr/local/bin/python
# encoding: utf-8
"""
Usage: python marketsim.py {-f} {-v} <starting cash> <order file> <output file portfolio value>

Simulate buys and sells in a market.
Takes starting cash and a csv file of orders and outputs the value of the portfolio for all days included in the order list

The csv file of orders looks like:

2008, 12, 3, AAPL, BUY, 130
2008, 12, 8, AAPL, SELL, 130
2008, 12, 5, IBM, BUY, 50

The outputted file of portfolio values looks like:

2008, 12, 3, 1000000
2008, 12, 4, 1000010
2008, 12, 5, 1000250

Created by Space on 2013-09-24.
Copyright (c) 2013. All rights reserved.
"""
import pdb

import sys
import math
import copy
import getopt
import datetime as dt
import os.path
import csv

import pandas as pd
import numpy as np

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


def enum(*sequential, **named):
    """
    create an enumeration type
    
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)
    
    
class CommonComparisonMixin(object):
    """
    mix-in the common comparison methods.  Requires __lt__ be defined for comparators other than equality
    
    """
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)
    
    def __gt__(self, other):
        return other.__le__(self)
        
    def __ge__(self, other):
        return other.__lt__(self)


class Order(CommonComparisonMixin):
    Actions = enum('BUY', 'SELL')
    
    def __init__(self, dt_day, stock, action, amount):
        self.dt_day = dt_day
        self.stock = stock
        if isinstance(action, str):
            action = Order.Actions.__dict__[action.upper()]
        self.action = action
        self.amount = int(amount)
    
    def __lt__(self, other):
        return isinstance(other, self.__class__) \
            and (self.dt_day < other.dt_day \
                 or (self.dt_day == other.dt_day \
                     and (self.action < other.action \
                          or (self.action == other.action \
                              and (self.stock < other.stock \
                                   or (self.stock == other.stock and self.amount < other.amount))))))
    
    def num_shares(self):
        """
        returns the signed number of shares associated with this order.  
        If this is a BUY, then this is a positive number.  If a SELL, a negative number
        """
        multiplier = (self.action == Order.Actions.BUY) and 1 or -1
        return multiplier * self.amount
        
    @classmethod
    def reader(cls, filename):
        with open(filename, "rbU") as f:
            for row in csv.reader(f):
                debug (", ".join(row))
                yield cls(dt.datetime(*map(int, row[0:3])), row[3],row[4],row[5])
    


def get_data(dt_start, dt_end, equities):
    if dt_start >= dt_end: raise ValueError("end date should be after start date")

    dataobj = da.DataAccess('Yahoo') # , cachestalltime=0)

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)    

    ldf_data =dataobj.get_data(ldt_timestamps, equities, 'close')

    # Filling the data for NAN
    ldf_data = ldf_data.fillna(method='ffill')
    ldf_data = ldf_data.fillna(method='bfill')
    ldf_data = ldf_data.fillna(1.0)

    return ldf_data
    
    
def execute(cash, orders):
    # don't muck with the order list passed in--make a copy
    orders = copy.copy(orders)
    orders.sort()
    portfolio = dict((o.stock,0) for o in orders)
    debug("executing orders of %s\n\tfrom %s to %s" % tuple([", ".join(portfolio.keys())] + [o.dt_day.strftime("%Y-%m-%d") for o in (orders[0],orders[-1])]))
    # add an extra day to make sure we include the day of the last order
    ldf_data = get_data(orders[0].dt_day, orders[-1].dt_day + dt.timedelta(days=1), portfolio.keys())
    orders.reverse()
    next_order = orders.pop()

    for row in ldf_data.iterrows():
        dt_day = row[0]
        while next_order.dt_day <= dt_day:
            num_shares = next_order.num_shares()
            cost = num_shares * row[1][next_order.stock]
            debug("%s: %s %d %s at %.2f (total %.2f)" % (next_order.dt_day.strftime("%Y-%m-%d"),
                                                         Order.Actions.reverse_mapping[next_order.action],
                                                         num_shares, next_order.stock, 
                                                         row[1][next_order.stock], cost))
            cash -= cost
            portfolio[next_order.stock] += num_shares
            if orders:
                next_order = orders.pop()
            else:
                next_order = None
                break
        yield (dt_day, cash + sum(row[1][portfolio.keys()] * portfolio.values()), portfolio)
    # there should be no more orders pending at the end
    assert(not next_order)
    
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
        
    cash = None
    orders_file = None
    out_file = None
    force = False
    global verbose
    
    # parse arguments
    try:
        opts, args = getopt.getopt(argv[1:], "?hvfc:o:", \
                                   ["help","force",
                                    "cash=","orders=","out="])
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
        elif option in ("-c","--cash"):
            cash = float(value)
        elif option in ("-o","--orders"):
            orders_file = value
        elif option in ("-out"):
            out_file = value
        else:
            raise Usage("unknown option: %s" % option)
    try:
        if cash is None:
            cash = float(args.pop(0))
        if orders_file is None:
            orders_file = args.pop(0)
        if out_file is None:
            out_file = args.pop(0)
    except IndexError:
        raise Usage("expected additional arguments")
    
    if not os.path.exists(orders_file) and orders_file.find(".") == -1:
        orders_file = orders_file + ".csv"
    if out_file.find(".") == -1:
        out_file = out_file + ".csv"

    if not force and os.path.exists(out_file):
        raise Usage("output file '%s' already exists. Add -f to force overwrite" % out_file)

    debug("Portfolio starting with $%.2f cash" % cash)
    debug("Reading orders from %s" % orders_file)
    debug("Writing values to %s" % out_file)
    debug()

    orders = [o for o in Order.reader(orders_file)]
    with open(out_file, "wb") as out:
        csvwriter = csv.writer(out)
        for (dt_day, value, portfolio) in execute(cash, orders):
            debug("%s: $%.2f invested in %s" % (dt_day.strftime("%Y-%m-%d"), value, ", ".join("%d %s" % (v, k) for (k,v) in portfolio.items())))
            csvwriter.writerow((dt_day.year, dt_day.month, dt_day.day, value))
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


