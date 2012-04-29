import numpy as np
import sys, os, re
from performance import *

if __name__ == "__main__":
    """Run main as a test, or to get the performance of a matching pair of files
    In the first case, don't give any command line arguments, in the second case
    give the ground_truth file as the first argument and the prediction_file
    as the second. It prints out the confusion matrix and the
    EER/Mean Recognition Rate
    
    """
    pattern = sys.argv[1]
    p = pattern.split('/')
    allfolders = os.listdir('/'.join(p[0:-1]))
    folders = ['/'.join(p[0:-1]+f) for f in allfolders \
        if re.search(p[-1], f) is not None]
        
    conf_matrices = None
    classlist = None
    eers = zeros([len(folders)])
    for i,folder in enumerate(folders):
        groundtruth_file = folder+'/gt.txt'
        prediction_file = folder+'/res.txt'
        print "Loading ground truth from: ", groundtruth_file
        groundtruth = []
        with open(groundtruth_file,'r') as f:
            # read the whole file, split it on newlines and filter out empty
            # lines (presumably at the end of the file)
            groundtruth = filter(None, f.read().split('\n'))
        #print groundtruth
        print "Loading prediction from: ", prediction_file
        prediction = []
        with open(prediction_file, 'r') as f:
            # read the whole file, split it on newlines and filter out empty
            # lines (presumably at the end of the file)
            prediction = filter(None, f.read().split('\n'))
        #print prediction
        confusion_matrix, cl = get_confusion_matrix(groundtruth, prediction)
        if conf_matrices == None:
            conf_matrices = np.zeros([len(cl),len(cl),len(folders)])
            classlist = cl
        conf_matrices[:,:,i] = confusion_matrix
        #print 'Confusion Matrix:'
        #print confusion_matrix
        #print 'Class list:', classlist
        eers[i] = get_equal_error_rate(confusion_matrix)
        print 'EER/MRR: ', eer
    print 'Summated Confusion Matrix:'
    print conf_matrices.sum(2)
    print 'Mean EER:', eers.mean()
    print 'Var EER:', eers.var()
    