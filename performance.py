from numpy import *
from ConfigParser import RawConfigParser


def get_confmat(c,ch):
    no_classes = max(c)+1
    confmat = zeros([no_classes, no_classes])
        
    for ci,chi in zip(c,ch):
        confmat[ci,chi] += 1
    return confmat

def get_equal_error_rate(confmat):
    ssums = confmat.sum(0)
    priors = ssums/ssums.sum()
    corrects = diag(confmat)
    
    truth_rates = corrects/ssums
    return (truth_rates*priors).sum()

def show_settings(results):
    config = results[0]
    print ' Settings:'
    for section in config.sections():
        if not section == 'Results':
            print '   [{0}]'.format(section)
            for option, value in config.items(section):
                print '     {0}: {1}'.format(option, value)

def get_results(results):

    for it, r in enumerate(results):
        c_hat = r.get('Results','classification')
        c_hat = [int(cc) for cc in c_hat[1:-1].split()]
        
        if it == 0:
            c = r.get('Results','ground_truth')
            c = [int(cc) for cc in c[1:-1].split()]
            
            c_hats = zeros([len(results), len(c_hat)])
        
        c_hats[it,:] = c_hat
    c_hat = c_hats.mean(0)
    return c, c_hat

def get_filenames(pattern):
    import os, re
    
    p = pattern[0].split('/')
    path = ''
    for pi in p[:-1]:
        path += pi+'/'
    files = os.listdir(path)
    patt = re.compile(p[-1])
    matches= [path+f for f in files if not isinstance(patt.search(f),type(None))]
    print matches
    return matches

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    parser.add_argument('-p','--pattern', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    if args.pattern:
        files = get_filenames(args.filename)
    else:
        files = args.filename
    
    results = []
    for f in files:
        rcp = RawConfigParser()
        rcp.read(f)
        results.append(rcp)
    print results
    
    print 'Showing results of {0}'.format(files)
    
    show_settings(results)
    
    c, c_hat = get_results(results)
    
    cf = get_confmat(c, c_hat)
    print 'Confusion matrix'
    print cf
    eer = get_equal_error_rate(cf)
    print 'ROC equal error rate: ', eer