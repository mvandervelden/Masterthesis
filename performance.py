#! /usr/bin/env python

from numpy import *
from ConfigParser import RawConfigParser
import sys, time

class Results(object):
    pass


def get_confmat(cs,chs):
    """ Returns a n-D confusion matrix of a ground truth classification vector c and a predicted classification ch.
        The rows determine the ground truth, the columns the predicted value. """
    # Get the number of classes
    no_classes = cs.max()+1
    # Initialize the matrix
    confmat = zeros([no_classes, no_classes])
    # Iterate over each pair of class-classification and add the combination to the entry of the confusion matrix
    for c,ch in zip(cs,chs):
        for i,j in zip(c,ch):
            confmat[i,j] += 1
    return confmat

def get_equal_error_rate(confmat):
    """ Calculate the n-D ROC Equal Error Rate of a confusion matrix"""
    
    # Get the prior probability of each class by dividing the amount of images in the class by the total
    ssums = confmat.sum(1)
    priors = ssums/ssums.sum()
    # Get the amount of "true positives" for each class (the diagonal of the CF)
    corrects = diag(confmat)
    # Get the true positive rate for each class by dividing the amounts of true positives by their amounts
    truth_rates = corrects/ssums
    print 'Truth rates:'
    print truth_rates
    # The sum of true positive rates times their prior over the classes defines the equal error rate
    # (with 2 classes, this is p1*tpr + p2*(1-fpr)[=tnr])
    return (truth_rates*priors).sum()

def show_settings(config):
    """ From a list of cfg-like result files 'results', print the values of the first file, except the 'results' section"""
    print ' Settings:'
    # Iterate over the sections
    for section in config.sections():
        # Except the results section
        if not section == 'Results':
            # Print the section name
            print '   [{0}]'.format(section)
            # Iterate through the items of the current section
            for option, value in config.items(section):
                # Print the item (name+ value)
                print '     {0}: {1}'.format(option, value)

def get_results(results, no_nns):
    """ Returns the results for all cPickle-results files """
    import cPickle as cpk
    
    res = [Results() for i in range(len(results))]
    # Iterate through the resultfiles
    for it, r in enumerate(results):
        # Open the resultfile and parse it as a cPickle file
        with open(r,'rb') as pkl:
            res[it].configfile = cpk.load(pkl)
            res[it].c = cpk.load(pkl)
            res[it].c_hat = cpk.load(pkl)
            res[it].trainfiles = cpk.load(pkl)
            res[it].testfiles = cpk.load(pkl)
            res[it].dssize = cpk.load(pkl)
            res[it].classlist = cpk.load(pkl)
            if not no_nns:
                res[it].nns = cpk.load(pkl)
    return res

def get_filenames(pattern):
    """ Function that finds all files that match a regex pattern. The pattern is assumed to be a path from the working
        working directory. It returns a list of the matching files"""
    import os, re
    
    matches = []
    for patt in pattern:
        # Break down the path into folders
        p = patt.split('/')
        # Determine the path from the list of folders p
        path = ''
        for folder in p[:-1]:
            path += folder+'/'
        # Get all files in the path
        files = os.listdir(path)
        # assume the last part of the pattern is a regular expression: compile it
        patt = re.compile(p[-1])
        # Add the matching files to the list of matches, if the pattern matches any files at all
        matches= matches+[path+f for f in files if not isinstance(patt.search(f),type(None))]
        print matches
    return matches

def visualize_features(r):
    """ Visualize the features/classification of a test image.
        This function assumes scipy (ndimage), PIL (Image), matplotlib (pylab) and Koen's DescriptorIO are available
        Besides: the test file's descriptors should be available"""
    import DescriptorIO
    import pylab as plb
    import Image
    import scipy.ndimage as nd
    testfile = r.testfiles[0]
    print 'Visualizing first testimage: {0}'.format(testfile)
    
    print r.samples
    features_by_scale = []
    all_features = zeros([0,5])
    for s in r.samples:
        f, _ = DescriptorIO.readDescriptors(s)
        features_by_scale.append(f)
        all_features = vstack([all_features, f])
    # Load the testimage, in grayscale ("L")
    image = Image.open(testfile).convert("L")
    im = asarray(image)
    A = zeros(im.shape)
    B = zeros(im.shape)
    C = zeros(im.shape)
    # Normalize the distances to be between 0 and 1, between the global min and max,
    # Furthermore, convert to a 'heat map' instead of a 'distance_map', so 1-dist
    dists = float_(r.nns)
    dists = 1-(dists-dists.min())/(dists.max()-dists.min())
    diff = dists[0]-dists[1]
    # For all feature centers, give A the similarity to class 0, B the similarity to class 1
    # Convert the feature indexes to ints
    features_by_scale = int_(features_by_scale[0])
    A[features_by_scale[:,1],features_by_scale[:,0]] = dists[0,:features_by_scale.shape[0]]
    B[features_by_scale[:,1],features_by_scale[:,0]] = dists[1,:features_by_scale.shape[0]]
    C[features_by_scale[:,1],features_by_scale[:,0]] = diff[:features_by_scale.shape[0]]
    # TODO: make the blur dependent on the scale of the features
    A_blur = nd.gaussian_filter(A,1.2*10)
    B_blur = nd.gaussian_filter(B,1.2*10)
    C_blur = nd.gaussian_filter(C,1.2*10)
    # plot the original image and their heat maps
    plb.subplot(2,2,1)
    plb.imshow(im,cmap=plb.cm.gray)
    plb.subplot(2,2,2)
    plb.imshow(A_blur)
    plb.colorbar()
    plb.subplot(2,2,3)
    plb.imshow(B_blur)
    plb.colorbar()
    plb.subplot(2,2,4)
    plb.imshow(C_blur)
    plb.colorbar()
    plb.show()

def timeit(f):
    """ Annotate a function with its elapsed execution time. """
    def timed_f(*args, **kwargs):
        t1 = time.time()
        
        try:
            res = f(*args, **kwargs)
        finally:
            t2 = time.time()
        
        timed_f.func_time = ((t2 - t1) / 60.0, t2 - t1, (t2 - t1) * 1000.0)
        
        if __debug__:
            sys.stdout.write("%s took %0.3fm %0.3fs %0.3fms\n" % (
                f.func_name,
                timed_f.func_time[0],
                timed_f.func_time[1],
                timed_f.func_time[2],
            ))
        return res
        
    return timed_f

if __name__ == '__main__':
    import argparse
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    parser.add_argument('-p','--pattern', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-x', '--visualize', action='store_true')
    # Debug flag, because it seems that that nns is often not saved due to crashes. This enables me to read the results anyway
    parser.add_argument('-n', '--no_nns', action='store_true')
    args = parser.parse_args()
    # if the -p flag is set, search for files conforming the given pattern(s), else assume a list of files/paths.
    # Extension is assumed to be .pkl (pickle), it will be added when necessary
    if args.pattern:
        results = get_filenames(args.filename)
    else:
        results = args.filename
    # iterate through the files given
    
    # get_results = timeit(get_results)
    # show_settings = timeit(show_settings)
    # Get the results
    res_list = get_results(results, args.no_nns)
    # Parse the settings file
    show_settings(res_list[0].configfile)
    # Show the settings of the test
    print 'Showing results of {0} files:\n\t{1}'.format(len(results),results)
    if not args.no_nns:
        print 'Classes used:',res_list[0].classlist
    c_hats = vstack([r.c_hat for r in res_list])
    cs = vstack([r.c for r in res_list])
    
    # Show no of errors (simple measure, check)
    print 'Hits: {0} out of {1}'.format((c_hats == cs).sum(), c_hats.shape[1])
    
    # create combined confusion matrix for all results files
    cf = get_confmat(cs, c_hats)
    print 'Confusion matrix'
    print cf
    print 'c_hats[0]'
    print c_hats[0]
    # Get equal error rate:
    eer = get_equal_error_rate(cf)
    print 'ROC equal error rate: ', eer
    
    if args.visualize:
        # Only do this on own computer, because matplotlib is required, unles you have x-window forwarding and matplotlib installed remotely
        visualize_features(res_list[0])
    
        
