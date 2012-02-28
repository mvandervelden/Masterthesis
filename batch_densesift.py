#!/usr/bin/env python
# encoding: utf-8
"""
batch_densesift.py

Created by Maarten van der Velden on 2012-02-21.
Copyright (c) 2012 UvA. All rights reserved.
"""

import argparse, os
from subprocess import call

def runBatch(args):
    path = args.data
    files = os.listdir(path)
    if not 'descriptors' in files:
        os.mkdir(path+'/descriptors')
    outputbase = path+"descriptors/"+args.detector + "_" + args.descriptor + "_"
    if args.binary:
        cdeopts = ["--detector", args.detector, "--descriptor", args.descriptor, "--outputFormat", "binary", "--output"]
        outputext = '.dbin'
    else:
        cdeopts = ["--detector", args.detector, "--descriptor", args.descriptor, "--output"]        
        outputext = '.dko'
    for f in files:
        if args.verbose:
            print "generating descriptors for %s"% f
        
        if f[-4:] == ".jpg":
            o = outputbase+f[:-4]+outputext
            run_args= ['colorDescriptor', path+f] + cdeopts + [o]
            #print run_args
            #print type(run_args)
            res = call(run_args)
            if res == 1:
                raise Exception("ColDescriptor run failed. Did not make output for %s" % f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-t', '--detector',default='densesampling')
    parser.add_argument('-s', '--descriptor', default='sift')
    parser.add_argument('-b', '--binary', action='store_true')
    args = parser.parse_args()
    
    if args.verbose:
        print 'data: ', args.data
        print 'detector: ', args.detector
        print 'descriptor: ', args.descriptor
        print 'binary: ', args.binary
    runBatch(args)
    