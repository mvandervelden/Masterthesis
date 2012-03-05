from numpy import *

def get_confmat(c,ch, no_classes):
    confmat = zeros([no_classes, no_classes])
        
    for ci,chi in zip(c,ch):
        confmat[ci,chi] += 1
    return confmat

def get_ROC_equal_rate(confmat):
    ssums = confmat.sum(0)
    priors = ssums/ssums.sum()
    corrects = diag(confmat)
    
    truth_rates = corrects/ssums
    return (truth_rates*priors).sum()

