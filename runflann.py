#!/usr/bin/env python
# encoding: utf-8
"""
runflann.py

Created by Maarten van der Velden on 2012-02-22.
Copyright (c) 2012 __MyCompanyName__. All rights reserved.
"""

import sys, argparse
import os, re

class Parameters():
    pass

def main(selection_params):
    train_set, test_set = selectData(selection_params)
    
    
def selectData(params):
    if params.data == 'graz01_person':
        pathp = "im/graz01/persons/descriptors/"
        pathb = "im/graz01/bikes/descriptors/"
        pathn = "im/graz01/no_bike_no_person/descriptors/"
        trsz = params.trainsize
        tesz = params.testsize
        pos_descr = getRandomDescriptors(pathp, params.fname, trsz + tesz)
        bike_descr = getRandomDescriptors(pathp, params.fname, (trsz + tesz)/2.)
        no_descr = getRandomDescriptors(pathp, params.fname, (trsz + tesz)/2.)
        train_set = [pos_descr[:trsz], bike_descr[:trsz/2.] + no_descr[:trsz/2.]]
        test_set = [pos_descr[trsz:], bike_descr[trsz/2.:] + no_descr[trsz/2.:]]
        return train_set, test_set
    elif params.data == 'graz01_bike':
        pass
    elif params.data == 'caltech101':
        pass
    
def getRandomDescriptors(path, fname, amount):
    files = os.listdir(path)
    def filt(x): return re.search('.cde',x)
    files = filter(filt,files)
    s = size(files)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-d', '--data')
    parser.add_argument('-t', '--detector',default='densesampling')
    parser.add_argument('-s', '--descriptor', default='sift')
    parser.add_argument('-n', '--trainsize', default=100)
    parser.add_argument('-m', '--testsize', default=100)
    args = parser.parse_args()
    sel_pars = Parameters()
    sel_pars.data = args.data
    sel_pars.fname = "%s_%s_" % args.detector, args.descriptor
    sel_pars.trainsize = args.trainsize
    sel_pars.testsize = args.testsize
    
    main(sel_pars)

