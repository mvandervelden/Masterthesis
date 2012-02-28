#!/usr/bin/env python
# encoding: utf-8
"""
boiman.py

Created by Maarten van der Velden on 2012-02-21.
Copyright (c) 2012 __MyCompanyName__. All rights reserved.
"""
print 'TEST!'

import sys, argparse

verbose = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-d', '--data')
    args = parser.parse_args()
    
    
    