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
    import re
    import cPickle as cpk
    for it, r in enumerate(results):
        with open(r,'rb') as pkl:
            c = cpk.load(pkl)
            c_hat = cpk.load(pkl)
            trainfiles = cpk.load(pkl)
            testfiles = cpk.load(pkl)
            nns = cpk.load(pkl)
            features = cpk.load(pkl)
            samples = cpk.load(pkl)
    return c, c_hat, trainfiles, testfiles, nns,features, samples

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

def gauss_kern(sigma):
    """ Returns a normalized 1D gauss kernel array for convolutions """
    # take the range of x to be from -3sigma tot 3sigma
    x = arange(-3*sigma,(3*sigma)+1)
    # Calculate the gaussian
    G = (sigma*sqrt(2*pi))**(-1)*exp(-(x**2)/(2*sigma**2))
    # Normalize the gaussian to ensure that the intensity remains the same.
    return G/G.sum()
    
def blur_image(im, sig, sigy=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    gx = gauss_kern(sig)
    if not sigy:
        sigy = sig
    gy = gauss_kern(sigy)
    print gx, gy
    for r in range(im.shape[0]):
        im[r,:] = convolve(im[r,:], gx, 'same')
    for c in range(im.shape[1]):
        im[:,c] = convolve(im[:,c], gy, 'same')
    return im


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    parser.add_argument('-p','--pattern', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-x', '--visualize', action='store_true')
    args = parser.parse_args()
    if args.pattern:
        files = get_filenames(args.filename)
    else:
        files = args.filename
    
    configs= []
    results=[]
    for f in files:
        rcp = RawConfigParser()
        rcp.read(f+'.res')
        configs.append(rcp)
        results.append(f+'.cpk')
    
    
    show_settings(configs)
    print 'Showing results of {0}'.format(files)
    
    c, c_hat, trainfiles, testfiles, nns, features, samples = get_results(results)
    
    cf = get_confmat(c, c_hat)
    print 'Confusion matrix'
    print cf
    eer = get_equal_error_rate(cf)
    print 'ROC equal error rate: ', eer

    if args.visualize:
        # Only do this on own computer, because matplotlib is requirde, unles you have x-window forwarding and matplotlib installed remotely
        print 'Visualizing first testimage: {0}'.format(testfiles[0])
        import DescriptorIO
        import matplotlib.pyplot as plt
        import matplotlib.image as mi   
                
        print samples
        features_by_scale = []
        all_features = zeros([0,5])
        for s in samples:
            f, _ = DescriptorIO.readDescriptors(s)
            features_by_scale.append(f)
            all_features = vstack([all_features, f])

        # clmin = argmin(nns,0)
        #dmin = min(nns,0)
        
        # clmax = clmin.max()
        # features_by_class = [array(0) for cl in range(clmax)]
        # dists_by_class
        # for cl in range(clmin.max()):
        #     features_by_class[cl] = all_features[clmin]
                
        im = mi.imread(testfiles[0])
        A = zeros(im.shape[0:2])
        B = zeros(im.shape[0:2])

        A[int_(features_by_scale[0][:,1]),int_(features_by_scale[0][:,0])] = nns[0,:features_by_scale[0].shape[0]]
        B[int_(features_by_scale[0][:,1]),int_(features_by_scale[0][:,0])] = nns[1,:features_by_scale[0].shape[0]]
        A *=1/A.max()
        
        plt.subplot(2,2,1)
        plt.imshow(im, origin='lower')
        #plt.plot(f[:,0],f[:,1],'go')
        plt.subplot(2,2,2)
        plt.imshow(A)
        plt.subplot(2,2,3)
        A_norm = blur_image(A,11)
        plt.imshow(A_norm)
        plt.subplot(2,2,4)
        B_norm = blur_image(B,11)
        plt.imshow(B_norm)
        plt.show()
        