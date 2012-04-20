import numpy as np
import random
import sys

def get_confusion_matrix(ground_truth, classification):
    """Function to create a confusion matrix from two lists of classes.
    
    Returns:
    An n-dimensional matrix, where n is the number of unique classes, and a list
    of the unique classes in the correct order. Each row is a true class, each
    column a predicted class.
    
    Keyword arguments:
    ground_truth -- a list of strings representing the true class names of each
        object.
    classification -- a list of strings representign the derived class names of 
        each object.
    
    """
    # get the list of unique classes
    class_list= list(set(ground_truth))
    class_list.sort()
    # Get the number of classes
    no_classes = len(class_list)
    # convert classnames to integers corresponding to the class_list
    class_dict = dict()
    for i,c in enumerate(class_list):
        class_dict[c]=i
    gt = [class_dict[g] for g in ground_truth]
    cf = [class_dict[c] for c in classification]
    # Initialize the matrix
    confmat = np.zeros([no_classes, no_classes])
    # Iterate over each pair of class-classification and add the combination to
    # the entry of the confusion matrix
    for g,c in zip(gt,cf):
            confmat[g,c] += 1
    return confmat, class_list

def get_equal_error_rate(confmat):
    """Calculate the n-D ROC Equal Error Rate (also called Mean Recognition Rate
    for n>2) of a confusion matrix.
    
    Returns:
    np.float64 value of the EER.
    
    Keyword arguments:
    confmat -- np.array confusion matrix. Each row is a true class, each column
        a predicted class.
    
    """
    # Get the prior probability of each class by dividing the amount of image
    # in the class by the total
    class_occurrences = confmat.sum(1)
    class_priors = class_occurrences / class_occurrences.sum()
    # Get the amount of "true positives" for each class (the diagonal of the CF)
    corrects_per_class = np.diag(confmat)
    # Get the true positive rate (recognition rate) for each class by dividing
    # the amounts of true positives by their amounts
    recognition_rates = corrects_per_class / class_occurrences
    print 'Truth rates per class:'
    print recognition_rates
    # The sum of true positive rates times their prior over the classes defines
    # the equal error rate (with 2 classes, this is p1*tpr + p2*(1-fpr)[=tnr])
    return (recognition_rates*class_priors).sum()

if __name__ == "__main__":
    """Run main as a test, or to get the performance of a matching pair of files
    In the first case, don't give any command line arguments, in the second case
    give the ground_truth file as the first argument and the prediction_file
    as the second. It prints out the confusion matrix and the
    EER/Mean Recognition Rate
    
    """
    if len(sys.argv) == 3:
        groundtruth_file = sys.argv[1]
        prediction_file = sys.argv[2]
        print "Loading ground truth from: ", groundtruth_file
        groundtruth = []
        with open(groundtruth_file,'r') as f:
            # read the whole file, split it on newlines and filter out empty
            # lines (presumably at the end of the file)
            groundtruth = filter(None, f.read().split('\n'))
        print groundtruth
        print "Loading prediction from: ", prediction_file
        prediction = []
        with open(prediction_file, 'r') as f:
            # read the whole file, split it on newlines and filter out empty
            # lines (presumably at the end of the file)
            prediction = filter(None, f.read().split('\n'))
        print prediction
    else:
        print "Running demo"
        groundtruth = ['a']*10+['b']*10+['c']*10
        print 'groundtruth: ', groundtruth
        prediction = groundtruth[:]
        random.shuffle(prediction)
        print 'prediction: ', prediction
    confusion_matrix, classlist = get_confusion_matrix(groundtruth, prediction)
    print 'Confusion Matrix:'
    print confusion_matrix
    print 'Class list:', classlist
    eer = get_equal_error_rate(confusion_matrix)
    print 'EER/MRR: ', eer
    