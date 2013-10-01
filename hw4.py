#!/usr/local/bin/python
# encoding: utf-8
"""
Usage: python hw4.py {-fv} --start=<date> {--end=<date>|--duration=<# days>} 
                     --dataset=(2008|2012) --threshold=<threshold>
                     --study.out=<filename> --study.view=(True|False) --study.compare=<index>
                     --trades.out=<filename> --trades.action=(BUY|SELL) --trades.amount=<# shares> --trades.after=<# days> 

A wrapper to call hw2.py, marketsim.py and analyze.py in sequence

Created by Space on 2013-09-24.
Copyright (c) 2013. All rights reserved.
"""

import sys

import analyze
import hw2
import marketsim


class Usage(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv

    compare_args = []
    cash_arg = "--cash=50000"
    out = None
    for arg in argv:
        if arg.startswith("--out="):
            out = arg[len("--out="):]
            dot_ndx = out.rfind('.')
            if dot_ndx != 1 and out[dot_ndx+1:] in ("pdf","csv"):
                out = out[0:dot_ndx]
            argv.remove(arg)
        elif arg.startswith("--cash="):
            cash_arg = arg
            argv.remove(arg)
        elif arg.startswith("--compare="):
            compare_args.append(arg)
            argv.remove(arg)
    if not out:
        raise Usage("specify an output file!")
    
    out_study = out + "_study.pdf"
    out_trades = out + "_trades.csv"
    out_results = out + "_results.csv"
    out_analysis = out + "_analysis.pdf"
    
    try:
        print "**Calling hw2..."
        print "****Creating ", out_study, " and ", out_trades
        args = argv + ["-fv", "--study.out="+out_study, "--trades.out="+out_trades ] + compare_args
        print "hw2 " + " ".join(args[1:])
        ret = hw2.main(args)
    except hw2.Usage, err:
        raise Usage(err)
    
    if ret == 0:
        try:
            print "**Calling marketsim..."
            print "****Creating ", out_results
            args = ["-fv", cash_arg, out_trades, out_results ]
            print "marketsim " + " ".join(args)
            ret = marketsim.main([argv[0]] + args)
        except marketsim.Usage, err:
            raise Usage(err)
    
    if ret == 0:
        try:
            print "**Calling analyze..."
            print "****Creating ", out_analysis
            args = ["-fv", "--portfolio=" + out_results, "--out=" + out_analysis] + compare_args
            print "analyze " + " ".join(args)
            ret = analyze.main([argv[0]] + args)
        except analyze.Usage, err:
            raise Usage(err)
    
    return ret


if __name__ == "__main__":
    ret = 1
    try:
        ret = main()
    except Usage, err:
        print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        print >> sys.stderr, "\t for help use --help"
        ret = 2
    sys.exit(ret)
